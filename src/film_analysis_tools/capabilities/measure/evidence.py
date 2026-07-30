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


@dataclass(frozen=True)
class HeterogeneityEvidence:
    """Slow spatial variation in grain level, and whether it is anchored to the screen."""

    envelope_ratio: float
    """Spread of local residual RMS across the frame, relative to its mean."""

    screen_anchored_correlation: float | None
    """Correlation of the low-frequency pattern with a second, unrelated source.

    ``None`` when no second source was supplied. A high value means the pattern sits in screen
    coordinates rather than in the picture — scanner fixed-pattern or dirty-gate, not grain.
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
    blur_radius: int = 12,
) -> HeterogeneityEvidence:
    """Slow spatial variation, with an optional cross-source test for screen anchoring.

    Temporal differencing cannot see a pattern that is identical in every frame — it cancels
    exactly. Detecting screen-anchored structure therefore needs a *spatial* comparison against
    unrelated content: if the same low-frequency pattern appears at the same screen coordinates in
    material with different pictures, it belongs to the scan rather than the scene.
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
        if other_source.shape[1:] != frames.shape[1:]:
            raise DataError(
                f"cross-source comparison needs matching frame geometry: "
                f"{other_source.shape[1:]} vs {frames.shape[1:]}"
            )
        # Compare the LOW-frequency structure, not the high-pass residue. Slow heterogeneity
        # lives in exactly the band a high-pass throws away; subtracting a blur would remove the
        # thing being looked for. Averaging over frames suppresses grain, and the two sources
        # carry different pictures, so shared low-frequency structure is screen-anchored.
        first = box_blur(frames.mean(axis=0), blur_radius)
        second = box_blur(np.asarray(other_source, dtype=np.float64).mean(axis=0), blur_radius)
        pattern_a = first - first.mean()
        pattern_b = second - second.mean()
        if pattern_a.std() > EPS and pattern_b.std() > EPS:
            correlation = float(np.corrcoef(pattern_a.ravel(), pattern_b.ravel())[0, 1])

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
    "AmplitudeEvidence",
    "AmplitudePoint",
    "DistributionEvidence",
    "HeterogeneityEvidence",
    "SpectrumEvidence",
    "TemporalEvidence",
    "amplitude_evidence",
    "distribution_evidence",
    "heterogeneity_evidence",
    "spectrum_evidence",
    "temporal_evidence",
]
