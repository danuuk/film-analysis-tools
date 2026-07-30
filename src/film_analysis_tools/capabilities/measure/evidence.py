"""Five kinds of grain evidence, produced independently of one another.

The legacy path measured all of these somewhere, spread across `analyze_patches`,
`grain_properties`, `slow_heterogeneity_contract`, `zone_analysis` and `refit_temporal_spectrum`,
each with its own patch selection and its own flags. Re-running one of them consistently with the
others meant remembering how the others had been invoked.

Here each producer takes the same windows and returns its own result. None depends on another's
output, so any one can be re-run, replaced, or disbelieved on its own — which is what makes a
disagreement between two of them informative rather than merely confusing.

The five:

* :func:`amplitude_evidence` — amplitude against level, the curve's raw material.
* :func:`spectrum_evidence` — 2D noise power and spatial autocorrelation: grain *structure*.
* :func:`distribution_evidence` — shape and tails, which say whether "sigma" means anything.
* :func:`heterogeneity_evidence` — slow spatial variation, and whether it is screen-anchored.
* :func:`temporal_evidence` — correlation across lags, and whether it can be believed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from film_analysis_tools.capabilities.measure.residual import (
    DEFAULT_LAGS,
    box_blur,
    extract,
)
from film_analysis_tools.capabilities.measure.windows import Window
from film_analysis_tools.core.errors import DataError

EPS = 1.0e-12


def _residuals(frames: np.ndarray) -> np.ndarray:
    """Lag-1 differences, scaled so their variance is the per-frame variance at rho = 0."""
    return (frames[1:] - frames[:-1]) / np.sqrt(2.0)


# ------------------------------------------------------------- 1. amplitude vs level


@dataclass(frozen=True)
class AmplitudePoint:
    level: float
    sigma: float
    rho: float
    trustworthy: bool
    sample_count: int
    band: str


@dataclass(frozen=True)
class AmplitudeEvidence:
    """Amplitude against level, one point per window, with each point's own trust."""

    points: tuple[AmplitudePoint, ...]

    @property
    def trusted(self) -> tuple[AmplitudePoint, ...]:
        return tuple(point for point in self.points if point.trustworthy)

    def level_range(self) -> tuple[float, float]:
        if not self.points:
            return (0.0, 0.0)
        levels = [point.level for point in self.points]
        return (min(levels), max(levels))

    def as_record(self) -> dict[str, Any]:
        low, high = self.level_range()
        return {
            "evidence": "amplitude_vs_level",
            "point_count": len(self.points),
            "trusted_count": len(self.trusted),
            "level_min": low,
            "level_max": high,
            "points": [
                {
                    "level": point.level,
                    "sigma": point.sigma,
                    "rho": point.rho,
                    "trustworthy": point.trustworthy,
                    "sample_count": point.sample_count,
                    "band": point.band,
                }
                for point in self.points
            ],
        }


def amplitude_evidence(
    frames: np.ndarray, windows: Sequence[Window], *, lags: tuple[int, ...] = DEFAULT_LAGS
) -> AmplitudeEvidence:
    """Per-window amplitude, correlation-corrected, each carrying whether it may be believed."""
    points: list[AmplitudePoint] = []
    for window in windows:
        estimate = extract(window.slice_of(frames), lags=lags)
        points.append(
            AmplitudePoint(
                level=window.level,
                sigma=estimate.sigma,
                rho=estimate.rho,
                trustworthy=estimate.correlation_trustworthy,
                sample_count=estimate.sample_count,
                band=window.band,
            )
        )
    return AmplitudeEvidence(points=tuple(points))


# --------------------------------------------------------- 2. 2D NPS / autocorrelation


@dataclass(frozen=True)
class SpectrumEvidence:
    """Radially averaged noise power and the spatial autocorrelation it implies."""

    frequencies: tuple[float, ...]
    """Cycles per pixel."""

    power: tuple[float, ...]
    autocorrelation: tuple[float, ...]
    """Normalised, by pixel lag from 1."""

    whiteness: float
    """Ratio of high-band to low-band power. 1.0 is spectrally flat, i.e. white."""

    sample_count: int

    @property
    def is_white(self) -> bool:
        """Whether the residual is spectrally flat, as un-resampled sensor grain should be."""
        return 0.8 <= self.whiteness <= 1.25

    def as_record(self) -> dict[str, Any]:
        return {
            "evidence": "spectrum",
            "frequencies": list(self.frequencies),
            "power": list(self.power),
            "autocorrelation": list(self.autocorrelation),
            "whiteness": self.whiteness,
            "is_white": self.is_white,
            "sample_count": self.sample_count,
        }


def spectrum_evidence(
    frames: np.ndarray, windows: Sequence[Window], *, bins: int = 24, max_lag: int = 8
) -> SpectrumEvidence:
    """Noise power spectrum and spatial autocorrelation of the temporal residual."""
    stacks = [_residuals(window.slice_of(frames)) for window in windows]
    if not stacks:
        raise DataError("spectrum evidence needs at least one window")

    accumulated: np.ndarray | None = None
    count = 0
    for stack in stacks:
        for residual_frame in stack:
            centred = residual_frame - residual_frame.mean()
            power = np.abs(np.fft.fft2(centred)) ** 2 / centred.size
            accumulated = power if accumulated is None else accumulated + power
            count += 1
    assert accumulated is not None
    mean_power = accumulated / max(count, 1)

    height, width = mean_power.shape
    v = np.fft.fftfreq(height)[:, None]
    u = np.fft.fftfreq(width)[None, :]
    radius = np.hypot(v, u)
    edges = np.linspace(0.0, 0.5, bins + 1)
    centres: list[float] = []
    values: list[float] = []
    for index in range(bins):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        if mask.any():
            centres.append(float((edges[index] + edges[index + 1]) / 2))
            values.append(float(mean_power[mask].mean()))

    # Autocorrelation follows from the same power spectrum (Wiener-Khinchin).
    correlation = np.real(np.fft.ifft2(mean_power))
    correlation = correlation / max(float(correlation[0, 0]), EPS)
    lags = [float(correlation[0, lag]) for lag in range(1, min(max_lag, width) + 1)]

    half = len(values) // 2
    low = float(np.mean(values[:half])) if half else 0.0
    high = float(np.mean(values[half:])) if half else 0.0
    whiteness = high / low if low > EPS else 0.0

    return SpectrumEvidence(
        frequencies=tuple(centres),
        power=tuple(values),
        autocorrelation=tuple(lags),
        whiteness=whiteness,
        sample_count=count,
    )


# ------------------------------------------------------------- 3. distribution / tails


@dataclass(frozen=True)
class DistributionEvidence:
    """Shape of the residual distribution — whether a single sigma describes it at all."""

    std: float
    skew: float
    excess_kurtosis: float
    quantiles: dict[str, float]
    tail_ratio: float
    """p99.9 over p50 of ``|residual|``. About 3.5 for a Gaussian."""

    sample_count: int

    @property
    def is_gaussian(self) -> bool:
        """Whether sigma summarises the distribution, or hides a heavy tail."""
        return abs(self.skew) < 0.3 and abs(self.excess_kurtosis) < 0.6

    def as_record(self) -> dict[str, Any]:
        return {
            "evidence": "distribution",
            "std": self.std,
            "skew": self.skew,
            "excess_kurtosis": self.excess_kurtosis,
            "quantiles": dict(self.quantiles),
            "tail_ratio": self.tail_ratio,
            "is_gaussian": self.is_gaussian,
            "sample_count": self.sample_count,
        }


def distribution_evidence(frames: np.ndarray, windows: Sequence[Window]) -> DistributionEvidence:
    """Moments and tails of the pooled residual."""
    if not windows:
        raise DataError("distribution evidence needs at least one window")
    pooled = np.concatenate([_residuals(window.slice_of(frames)).ravel() for window in windows])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        raise DataError("distribution evidence needs finite residual samples")

    centred = pooled - pooled.mean()
    std = float(np.std(centred))
    if std <= EPS:
        raise DataError("residual has no variance; nothing to describe")

    normalised = centred / std
    magnitude = np.abs(pooled)
    return DistributionEvidence(
        std=std,
        skew=float(np.mean(normalised**3)),
        excess_kurtosis=float(np.mean(normalised**4) - 3.0),
        quantiles={
            str(percent): float(np.percentile(pooled, percent))
            for percent in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)
        },
        tail_ratio=float(np.percentile(magnitude, 99.9) / max(np.percentile(magnitude, 50), EPS)),
        sample_count=int(pooled.size),
    )


# ------------------------------------------------------ 4. slow spatial heterogeneity
#
# Two different questions live here, and conflating them was a real defect.
#
#   "Does a fixed pattern sit on top of every frame?"      -> additive_pattern_evidence
#   "Does a fixed envelope modulate the grain amplitude?"  -> heterogeneity_evidence
#
# The first is scanner shading, vignetting, dirt, fixed-pattern offsets. It lives in the temporal
# *mean*. The second is grain heterogeneity, and it is invisible in the temporal mean by
# construction: a zero-mean grain field multiplied by an envelope still averages to zero.
#
# Measured on two unrelated sequences sharing one multiplicative envelope, correlating blurred
# temporal means reported **-0.07** and declared the envelope not screen-anchored. The same
# detector on a shared *additive* pattern reports **0.98**. It was answering the additive question
# almost perfectly and the multiplicative question not at all.


@dataclass(frozen=True)
class AdditivePatternEvidence:
    """A fixed image added to every frame: scanner shading, vignetting, dirt, gate weave residue.

    Detected in the temporal mean, which is where an additive pattern survives and where a grain
    envelope does not. This says nothing about grain amplitude — see :class:`HeterogeneityEvidence`
    for that question.
    """

    cross_source_correlation: float | None
    """Correlation of the blurred temporal mean with a second, unrelated source.

    ``None`` when no second source was supplied. High means the same pattern sits at the same
    screen coordinates in different pictures, so it belongs to the scan.
    """

    sample_count: int

    @property
    def is_screen_anchored(self) -> bool | None:
        if self.cross_source_correlation is None:
            return None
        return self.cross_source_correlation > 0.3

    def as_record(self) -> dict[str, Any]:
        return {
            "evidence": "additive_fixed_pattern",
            "cross_source_correlation": self.cross_source_correlation,
            "is_screen_anchored": self.is_screen_anchored,
            "sample_count": self.sample_count,
        }


def additive_pattern_evidence(
    frames: np.ndarray,
    *,
    other_source: np.ndarray | None = None,
    blur_radius: int = 12,
) -> AdditivePatternEvidence:
    """Shared additive structure between two sources with unrelated pictures.

    Averaging over frames suppresses grain and leaves whatever is fixed. If the same low-frequency
    pattern appears at the same screen coordinates in different material, it is in the scan.
    """
    stack = np.asarray(frames, dtype=np.float64)
    correlation: float | None = None
    if other_source is not None:
        other = np.asarray(other_source, dtype=np.float64)
        if other.shape[1:] != stack.shape[1:]:
            raise DataError(
                f"cross-source comparison needs matching frame geometry: "
                f"{other.shape[1:]} vs {stack.shape[1:]}"
            )
        first = box_blur(stack.mean(axis=0), blur_radius)
        second = box_blur(other.mean(axis=0), blur_radius)
        correlation = _correlate(first - first.mean(), second - second.mean())
    return AdditivePatternEvidence(
        cross_source_correlation=correlation, sample_count=int(stack.size)
    )


def _correlate(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.std() <= EPS or second.std() <= EPS:
        return None
    return float(np.corrcoef(first.ravel(), second.ravel())[0, 1])


def grain_energy_map(
    frames: np.ndarray,
    *,
    blur_radius: int = 8,
    normalise_for_level: bool = True,
    level_bins: int = 12,
) -> np.ndarray:
    """Local grain energy per pixel, smoothed, with the amplitude-versus-level trend divided out.

    This is the quantity a grain envelope actually modulates. Built as:

    1. aligned temporal residuals (lag-1 differences, so a static picture cancels);
    2. local RMS over time, which is a noisy per-pixel estimate;
    3. division by the expected amplitude at that pixel's own level, so the ordinary
       amplitude-versus-level relationship is not mistaken for a spatial envelope;
    4. smoothing, because the envelope of interest is low-frequency.

    Step 3 matters most. Without it a bright object and a dark one differ in grain energy simply
    because grain amplitude depends on level, and that difference would read as heterogeneity in
    every source — including sources that share nothing.

    It also has a cost, and it is worth stating. When a fixed *additive* pattern happens to be
    collinear with the grain envelope — the shading and the amplitude modulation having the same
    shape, as with some vignetting — level normalisation attributes the energy variation to level
    and the envelope correlation collapses (measured: 0.98 to 0.20 on such a case). That is the
    correct answer to "is this variation explained by level", but it is not the same question as
    "is there an envelope". Pass ``normalise_for_level=False`` to ask the second question directly,
    and read it alongside :func:`additive_pattern_evidence` rather than on its own.
    """
    stack = np.asarray(frames, dtype=np.float64)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise DataError(f"grain energy needs at least two frames of (h, w); got {stack.shape}")

    residual = _residuals(stack)
    energy = np.sqrt(np.mean(residual**2, axis=0))
    energy = box_blur(energy, blur_radius)

    if normalise_for_level:
        level = box_blur(stack.mean(axis=0), blur_radius)
        expected = _expected_energy_at_level(level, energy, bins=level_bins)
        energy = energy / np.maximum(expected, EPS)
    return energy


def _expected_energy_at_level(level: np.ndarray, energy: np.ndarray, *, bins: int) -> np.ndarray:
    """Median grain energy as a function of level, interpolated back onto every pixel.

    Binned and interpolated rather than fitted to a shape: the amplitude-versus-level relationship
    is what the whole measurement exists to discover, so assuming a form for it here would be
    circular.
    """
    flat_level, flat_energy = level.ravel(), energy.ravel()
    edges = np.quantile(flat_level, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return np.full_like(energy, float(np.median(flat_energy)))

    centres, medians = [], []
    index = np.clip(np.searchsorted(edges, flat_level, side="right") - 1, 0, edges.size - 2)
    for bin_number in range(edges.size - 1):
        selected = flat_energy[index == bin_number]
        if selected.size:
            centres.append(0.5 * (edges[bin_number] + edges[bin_number + 1]))
            medians.append(float(np.median(selected)))
    if len(centres) < 2:
        return np.full_like(energy, float(np.median(flat_energy)))
    return np.interp(level, np.asarray(centres), np.asarray(medians))


@dataclass(frozen=True)
class HeterogeneityEvidence:
    """A fixed spatial envelope modulating grain amplitude, and whether it is screen-anchored."""

    envelope_ratio: float
    """Spread of local residual RMS across the frame, relative to its mean."""

    screen_anchored_correlation: float | None
    """Correlation of the level-normalised *grain energy* map with a second, unrelated source.

    ``None`` when no second source was supplied. A high value means grain amplitude is modulated
    by the same pattern at the same screen coordinates in different pictures, so the envelope
    belongs to the scan or the optics rather than to the negative.
    """

    sample_count: int

    @property
    def is_screen_anchored(self) -> bool | None:
        if self.screen_anchored_correlation is None:
            return None
        return self.screen_anchored_correlation > 0.3

    @property
    def belongs_in_a_negative_model(self) -> bool | None:
        """Screen-anchored structure is a property of the scan, not of the negative.

        It may still be worth reproducing as an optional appearance layer. It should not enter a
        negative model automatically, and this is the check that separates the two — which the
        legacy path could not do, because it never compared across sources.
        """
        anchored = self.is_screen_anchored
        return None if anchored is None else not anchored

    def as_record(self) -> dict[str, Any]:
        return {
            "evidence": "slow_heterogeneity",
            "envelope_ratio": self.envelope_ratio,
            "screen_anchored_correlation": self.screen_anchored_correlation,
            "is_screen_anchored": self.is_screen_anchored,
            "belongs_in_a_negative_model": self.belongs_in_a_negative_model,
            "sample_count": self.sample_count,
        }


def heterogeneity_evidence(
    frames: np.ndarray,
    windows: Sequence[Window],
    *,
    other_source: np.ndarray | None = None,
    blur_radius: int = 8,
    normalise_for_level: bool = True,
) -> HeterogeneityEvidence:
    """Spatial variation in grain *amplitude*, with a cross-source test for screen anchoring.

    The cross-source test compares level-normalised grain-energy maps, not temporal means. A grain
    envelope is multiplicative on a zero-mean field, so it leaves the temporal mean untouched and
    a mean-image comparison cannot see it — measured at **-0.07** on two unrelated sequences built
    with an identical envelope, against **0.98** for the same detector on a shared additive
    pattern. Comparing energy maps answers the question actually being asked.

    For the additive question — scanner shading, vignetting, dirt — use
    :func:`additive_pattern_evidence`, which is the old behaviour under an accurate name.
    """
    levels: list[float] = []
    total = 0
    for window in windows:
        residual_stack = _residuals(window.slice_of(frames))
        levels.append(float(np.sqrt(np.mean(residual_stack**2))))
        total += int(residual_stack.size)
    if not levels:
        raise DataError("heterogeneity evidence needs at least one window")

    mean_level = float(np.mean(levels))
    envelope_ratio = float(np.std(levels) / mean_level) if mean_level > EPS else 0.0

    correlation: float | None = None
    if other_source is not None:
        other = np.asarray(other_source, dtype=np.float64)
        if other.shape[1:] != frames.shape[1:]:
            raise DataError(
                f"cross-source comparison needs matching frame geometry: "
                f"{other.shape[1:]} vs {frames.shape[1:]}"
            )
        first = grain_energy_map(
            frames, blur_radius=blur_radius, normalise_for_level=normalise_for_level
        )
        second = grain_energy_map(
            other, blur_radius=blur_radius, normalise_for_level=normalise_for_level
        )
        correlation = _correlate(first - first.mean(), second - second.mean())

    return HeterogeneityEvidence(
        envelope_ratio=envelope_ratio,
        screen_anchored_correlation=correlation,
        sample_count=total,
    )


# ------------------------------------------------------------------ 5. temporal behaviour


@dataclass(frozen=True)
class TemporalEvidence:
    """Correlation across lags, and whether the windows agree about it."""

    rho: float
    rho_from_lag4: float
    lag_variances: dict[int, float]
    per_window_rho: tuple[float, ...]
    trusted_fraction: float
    """Share of windows whose correlation may be believed — the rest are drifting."""

    sample_count: int

    @property
    def independence_established(self) -> bool:
        """Whether temporal independence is *demonstrated*, not merely unrefuted.

        Requires correlation near zero, agreement between lags, and a majority of windows able to
        vouch for their own estimate. A drifting window reports near-zero correlation whatever the
        truth, so "rho looks small" is not evidence on its own.
        """
        return (
            abs(self.rho) < 0.1
            and abs(self.rho - self.rho_from_lag4) < 0.1
            and self.trusted_fraction >= 0.5
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "evidence": "temporal",
            "rho": self.rho,
            "rho_from_lag4": self.rho_from_lag4,
            "lag_variances": {str(lag): value for lag, value in self.lag_variances.items()},
            "per_window_rho": list(self.per_window_rho),
            "trusted_fraction": self.trusted_fraction,
            "independence_established": self.independence_established,
            "sample_count": self.sample_count,
        }


def temporal_evidence(
    frames: np.ndarray, windows: Sequence[Window], *, lags: tuple[int, ...] = DEFAULT_LAGS
) -> TemporalEvidence:
    """Temporal correlation pooled across windows, with per-window agreement."""
    estimates = [extract(window.slice_of(frames), lags=lags) for window in windows]
    if not estimates:
        raise DataError("temporal evidence needs at least one window")

    trusted = [estimate for estimate in estimates if estimate.correlation_trustworthy]
    basis = trusted or estimates
    pooled_variances: dict[int, float] = {}
    for lag in sorted(lags):
        values = [estimate.lag_variances.get(lag, 0.0) for estimate in basis]
        pooled_variances[lag] = float(np.mean(values))

    return TemporalEvidence(
        rho=float(np.median([estimate.rho for estimate in basis])),
        rho_from_lag4=float(np.median([estimate.rho_from_lag4 for estimate in basis])),
        lag_variances=pooled_variances,
        per_window_rho=tuple(estimate.rho for estimate in estimates),
        trusted_fraction=len(trusted) / len(estimates),
        sample_count=sum(estimate.sample_count for estimate in estimates),
    )


__all__ = [
    "AdditivePatternEvidence",
    "AmplitudeEvidence",
    "AmplitudePoint",
    "DistributionEvidence",
    "HeterogeneityEvidence",
    "SpectrumEvidence",
    "TemporalEvidence",
    "additive_pattern_evidence",
    "amplitude_evidence",
    "distribution_evidence",
    "grain_energy_map",
    "heterogeneity_evidence",
    "spectrum_evidence",
    "temporal_evidence",
]
