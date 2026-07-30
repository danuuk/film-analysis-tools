"""Spatial grain footprint: is the normalised shape stable enough for one common kernel?

This answers **one** question, the next after amplitude: once predicted amplitude is divided out,
is the spatial grain shape sufficiently stable across level and interval to use a single footprint,
or does it change with level enough to need a level-dependent kernel?

The method, in the order the answer depends on it:

1. Each window's residual is normalised by its **leave-one-interval-out** amplitude prediction, not
   the fit trained on its own interval -- otherwise the amplitude fit would absorb, and hide, a
   genuine interval-to-interval difference.
2. The residual is the aligned lag-one difference with the small correlation term kept exact:

       R_t = (I_t - I_{t-1}) / (sqrt(2 (1 - rho)) * sigma_hat(L))

   At rho ~ 0.005 the correction is numerically tiny, but the definition stays exact.
3. A normalised radial power spectrum is computed **per window, before pooling** -- pooling
   differently-scaled residuals and then taking one spectrum would smear the very shape in question.
4. The window is the measurement; the **interval** is the independent unit. Stability is judged by
   comparing between-level variation against the split-half and between-interval variation that are
   present even when the footprint is genuinely constant.

A single footprint is adopted only when between-level differences are no larger than that ordinary
variation. The distribution *shape* of shadow residuals is explicitly out of scope here: shadow
windows may contribute spectral evidence, but their heavy tails are not the generator's particle
law and must not be read as one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from film_analysis_tools.capabilities.measure.residual import aligned_residuals
from film_analysis_tools.core.errors import DataError

EPS = 1.0e-12

#: Frequency band, in cycles per pixel, over which spectra are compared. DC and the highest
#: frequencies are excluded: DC carries no shape, and the top bin is dominated by aliasing.
COMPARE_BAND: tuple[float, float] = (0.05, 0.45)

#: Codec block sizes whose fundamental frequency (1/size cycles/pixel) is checked for a peak.
BLOCK_SIZES: tuple[int, ...] = (8, 16)


def normalise_residual(frames: np.ndarray, sigma_hat: float, rho: float = 0.0) -> np.ndarray:
    """Aligned lag-one residual scaled to unit variance under the amplitude model.

    ``sigma_hat`` is the **predicted** amplitude at this window's level, supplied by the caller so
    the leave-one-interval-out discipline is visible at the call site rather than hidden here.
    """
    residual = aligned_residuals(frames)  # already divided by sqrt(2)
    scale = max(sigma_hat, EPS) * np.sqrt(max(1.0 - rho, EPS))
    return np.asarray(residual / scale)


def _radial_psd(residual: np.ndarray, bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum of a residual stack, normalised to unit mean in-band.

    Normalised to a *shape*: amplitude has already been divided out, and what remains of interest
    is how power is distributed across frequency, not its total.
    """
    power: np.ndarray | None = None
    for frame in residual:
        centred = frame - frame.mean()
        spectrum = np.abs(np.fft.fft2(centred)) ** 2 / centred.size
        power = spectrum if power is None else power + spectrum
    assert power is not None
    power = power / residual.shape[0]

    height, width = power.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.hypot(fy, fx)
    edges = np.linspace(0.0, 0.5, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    profile = np.zeros(bins)
    for index in range(bins):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        profile[index] = float(power[mask].mean()) if mask.any() else np.nan
    return centres, profile


def _in_band(centres: np.ndarray) -> np.ndarray:
    return (centres >= COMPARE_BAND[0]) & (centres <= COMPARE_BAND[1])


def _normalise_profile(centres: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Scale the profile to unit mean over the comparison band, so only its shape remains."""
    band = _in_band(centres) & np.isfinite(profile)
    reference = np.nanmean(profile[band]) if band.any() else np.nan
    return profile / reference if reference and np.isfinite(reference) else profile


def _grain_radius(residual: np.ndarray) -> float:
    """Half-width of the spatial autocorrelation, in pixels: a characteristic grain size.

    White grain has an autocorrelation that is a spike at zero lag, so the half-width is ~0.5 px;
    a spatially correlated (larger) grain widens it.
    """
    power: np.ndarray | None = None
    for frame in residual:
        centred = frame - frame.mean()
        spectrum = np.abs(np.fft.fft2(centred)) ** 2
        power = spectrum if power is None else power + spectrum
    assert power is not None
    autocorr = np.fft.ifft2(power).real
    autocorr = autocorr / max(autocorr[0, 0], EPS)
    line = autocorr[0, : autocorr.shape[1] // 2]
    below = np.where(line < 0.5)[0]
    if below.size == 0:
        return float(line.size)
    first = int(below[0])
    if first == 0:
        return 0.5
    # Linear interpolation to the 0.5 crossing between the two straddling lags.
    high, low = line[first - 1], line[first]
    frac = (high - 0.5) / max(high - low, EPS)
    return float(first - 1 + frac)


def _anisotropy(residual: np.ndarray) -> float:
    """Ratio of horizontal to vertical autocorrelation half-width. 1.0 is isotropic."""
    power: np.ndarray | None = None
    for frame in residual:
        centred = frame - frame.mean()
        power_frame = np.abs(np.fft.fft2(centred)) ** 2
        power = power_frame if power is None else power + power_frame
    assert power is not None
    autocorr = np.fft.ifft2(power).real
    autocorr = autocorr / max(autocorr[0, 0], EPS)

    def half_width(line: np.ndarray) -> float:
        below = np.where(line < 0.5)[0]
        return float(below[0]) if below.size else float(line.size)

    horizontal = half_width(autocorr[0, : autocorr.shape[1] // 2])
    vertical = half_width(autocorr[: autocorr.shape[0] // 2, 0])
    return horizontal / max(vertical, EPS)


def _block_peak(centres: np.ndarray, profile: np.ndarray) -> float:
    """Largest excess at a codec block fundamental over the local spectral background.

    A value near 1 means no block structure; well above 1 means the encoder's block grid is
    printing a periodic pattern into the residual.
    """
    peak = 1.0
    for size in BLOCK_SIZES:
        target = 1.0 / size
        index = int(np.argmin(np.abs(centres - target)))
        neighbours = [i for i in (index - 2, index + 2) if 0 <= i < profile.size]
        local = np.nanmean([profile[i] for i in neighbours]) if neighbours else np.nan
        if np.isfinite(local) and local > EPS and np.isfinite(profile[index]):
            peak = max(peak, float(profile[index] / local))
    return peak


@dataclass(frozen=True)
class WindowSpectrum:
    """One window's normalised spatial evidence, tagged with its independent unit."""

    interval: str
    level: float
    band: str
    frequencies: tuple[float, ...]
    radial_psd: tuple[float, ...]
    grain_radius: float
    anisotropy: float
    block_peak: float

    def as_record(self) -> dict[str, Any]:
        return {
            "interval": self.interval,
            "level": self.level,
            "band": self.band,
            "grain_radius": self.grain_radius,
            "anisotropy": self.anisotropy,
            "block_peak": self.block_peak,
            "radial_psd": list(self.radial_psd),
        }


def window_spectrum(
    frames: np.ndarray,
    *,
    interval: str,
    level: float,
    band: str,
    sigma_hat: float,
    rho: float = 0.0,
    bins: int = 24,
) -> WindowSpectrum:
    """Normalise one window's residual and reduce it to its spatial spectrum and shape metrics."""
    residual = normalise_residual(frames, sigma_hat, rho)
    if residual.ndim != 3 or residual.shape[0] < 1:
        raise DataError(f"window spectrum needs a residual stack (n, h, w); got {residual.shape}")
    centres, profile = _radial_psd(residual, bins=bins)
    normalised = _normalise_profile(centres, profile)
    return WindowSpectrum(
        interval=interval,
        level=level,
        band=band,
        frequencies=tuple(float(x) for x in centres),
        radial_psd=tuple(float(x) for x in normalised),
        grain_radius=_grain_radius(residual),
        anisotropy=_anisotropy(residual),
        block_peak=_block_peak(centres, normalised),
    )


def spectral_distance(a: WindowSpectrum, b: WindowSpectrum) -> float:
    """Log-spectral distance between two normalised radial profiles over the comparison band."""
    centres = np.asarray(a.frequencies)
    band = _in_band(centres)
    first = np.log(np.maximum(np.asarray(a.radial_psd)[band], EPS))
    second = np.log(np.maximum(np.asarray(b.radial_psd)[band], EPS))
    finite = np.isfinite(first) & np.isfinite(second)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean((first[finite] - second[finite]) ** 2)))


def _mean_spectrum(spectra: Sequence[WindowSpectrum]) -> WindowSpectrum:
    """The mean normalised profile over a group, kept as a WindowSpectrum for distance reuse."""
    stacked = np.array([s.radial_psd for s in spectra])
    mean_profile = np.nanmean(stacked, axis=0)
    first = spectra[0]
    return WindowSpectrum(
        interval="mean",
        level=float(np.mean([s.level for s in spectra])),
        band=first.band,
        frequencies=first.frequencies,
        radial_psd=tuple(float(x) for x in mean_profile),
        grain_radius=float(np.mean([s.grain_radius for s in spectra])),
        anisotropy=float(np.mean([s.anisotropy for s in spectra])),
        block_peak=float(np.mean([s.block_peak for s in spectra])),
    )


@dataclass(frozen=True)
class Variation:
    """The distances observed for one kind of comparison."""

    kind: str
    distances: tuple[float, ...]

    @property
    def median(self) -> float:
        return float(np.median(self.distances)) if self.distances else float("nan")

    @property
    def p90(self) -> float:
        return float(np.percentile(self.distances, 90)) if self.distances else float("nan")

    def as_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n": len(self.distances),
            "median": self.median,
            "p90": self.p90,
        }


@dataclass(frozen=True)
class BandSummary:
    """Shape metrics for one level band, with its independent-span count."""

    band: str
    windows: int
    intervals: int
    grain_radius: float
    anisotropy: float
    block_peak: float
    level_min: float
    level_max: float

    def as_record(self) -> dict[str, Any]:
        return {
            "band": self.band,
            "windows": self.windows,
            "intervals": self.intervals,
            "grain_radius": self.grain_radius,
            "anisotropy": self.anisotropy,
            "block_peak": self.block_peak,
            "level_min": self.level_min,
            "level_max": self.level_max,
        }

    def line(self) -> str:
        return (
            f"    {self.band:9s} {self.windows:3d}w/{self.intervals}iv  "
            f"grain_radius {self.grain_radius:.2f}px  anisotropy {self.anisotropy:.2f}  "
            f"block_peak {self.block_peak:.2f}  level {self.level_min:.4f}..{self.level_max:.4f}"
        )


def _band_summary(spectra: Sequence[WindowSpectrum]) -> BandSummary:
    return BandSummary(
        band=spectra[0].band,
        windows=len(spectra),
        intervals=len({s.interval for s in spectra}),
        grain_radius=float(np.median([s.grain_radius for s in spectra])),
        anisotropy=float(np.median([s.anisotropy for s in spectra])),
        block_peak=float(np.median([s.block_peak for s in spectra])),
        level_min=float(min(s.level for s in spectra)),
        level_max=float(max(s.level for s in spectra)),
    )


@dataclass(frozen=True)
class FootprintStability:
    """Whether one common footprint suffices, judged against the natural variation."""

    split_half: Variation
    between_interval: Variation
    between_level: Variation
    bands_present: tuple[str, ...]
    bands: tuple[BandSummary, ...] = ()

    @property
    def one_footprint_suffices(self) -> bool | None:
        """True when between-level variation is no larger than the ordinary variation.

        The null is what the same footprint looks like split two ways or seen in two intervals.
        A between-level difference within that band is not evidence of a level trend. ``None`` when
        there are too few level bands to compare -- unanswerable, not answered no.
        """
        if len(self.bands_present) < 2 or not self.between_level.distances:
            return None
        baseline = max(self.split_half.p90, self.between_interval.p90)
        return self.between_level.median <= baseline

    def as_record(self) -> dict[str, Any]:
        return {
            "bands_present": list(self.bands_present),
            "bands": [band.as_record() for band in self.bands],
            "split_half": self.split_half.as_record(),
            "between_interval": self.between_interval.as_record(),
            "between_level": self.between_level.as_record(),
            "one_footprint_suffices": self.one_footprint_suffices,
        }

    def summary(self) -> str:
        verdict = {True: "ONE footprint suffices", False: "level-dependent footprint indicated"}
        lines = [
            "spatial footprint stability (log-spectral distance):",
            f"  split-half     median {self.split_half.median:.3f}  p90 {self.split_half.p90:.3f}"
            f"  (n={len(self.split_half.distances)})",
            f"  between-interval median {self.between_interval.median:.3f}  "
            f"p90 {self.between_interval.p90:.3f}  (n={len(self.between_interval.distances)})",
            f"  between-level  median {self.between_level.median:.3f}  "
            f"p90 {self.between_level.p90:.3f}  (n={len(self.between_level.distances)})",
        ]
        for band in self.bands:
            lines.append(band.line())
        answer = self.one_footprint_suffices
        if answer is None:
            lines.append("  verdict: cannot assess -- fewer than two level bands present")
        else:
            lines.append(f"  verdict: {verdict[answer]}")
        return "\n".join(lines)


def _split_half_distances(spectra: Sequence[WindowSpectrum], *, seed: int = 0) -> list[float]:
    """Distances between two random halves of the same (interval, band) group."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[WindowSpectrum]] = {}
    for spectrum in spectra:
        groups.setdefault((spectrum.interval, spectrum.band), []).append(spectrum)
    distances: list[float] = []
    for members in groups.values():
        if len(members) < 4:
            continue
        order = rng.permutation(len(members))
        half = len(members) // 2
        left = [members[i] for i in order[:half]]
        right = [members[i] for i in order[half : 2 * half]]
        distances.append(spectral_distance(_mean_spectrum(left), _mean_spectrum(right)))
    return distances


def _between_interval_distances(spectra: Sequence[WindowSpectrum]) -> list[float]:
    """Distances between the mean spectra of different intervals, within the same band."""
    by_band: dict[str, dict[str, list[WindowSpectrum]]] = {}
    for spectrum in spectra:
        by_band.setdefault(spectrum.band, {}).setdefault(spectrum.interval, []).append(spectrum)
    distances: list[float] = []
    for intervals in by_band.values():
        means = {i: _mean_spectrum(m) for i, m in intervals.items() if len(m) >= 2}
        keys = sorted(means)
        for left in range(len(keys)):
            for right in range(left + 1, len(keys)):
                distances.append(spectral_distance(means[keys[left]], means[keys[right]]))
    return distances


def _between_level_distances(spectra: Sequence[WindowSpectrum]) -> list[float]:
    """Distances between the mean spectra of different bands, within the same interval."""
    by_interval: dict[str, dict[str, list[WindowSpectrum]]] = {}
    for spectrum in spectra:
        by_interval.setdefault(spectrum.interval, {}).setdefault(spectrum.band, []).append(spectrum)
    distances: list[float] = []
    for bands in by_interval.values():
        means = {b: _mean_spectrum(m) for b, m in bands.items() if len(m) >= 2}
        keys = sorted(means)
        for left in range(len(keys)):
            for right in range(left + 1, len(keys)):
                distances.append(spectral_distance(means[keys[left]], means[keys[right]]))
    return distances


def assess_stability(spectra: Sequence[WindowSpectrum], *, seed: int = 0) -> FootprintStability:
    """Run the split-half, between-interval and between-level comparisons."""
    if len(spectra) < 4:
        raise DataError(f"need at least 4 window spectra to assess stability; got {len(spectra)}")
    by_band: dict[str, list[WindowSpectrum]] = {}
    for spectrum in spectra:
        by_band.setdefault(spectrum.band, []).append(spectrum)
    return FootprintStability(
        split_half=Variation("split_half", tuple(_split_half_distances(spectra, seed=seed))),
        between_interval=Variation("between_interval", tuple(_between_interval_distances(spectra))),
        between_level=Variation("between_level", tuple(_between_level_distances(spectra))),
        bands_present=tuple(sorted(by_band)),
        bands=tuple(_band_summary(members) for _, members in sorted(by_band.items())),
    )


__all__ = [
    "BLOCK_SIZES",
    "COMPARE_BAND",
    "BandSummary",
    "FootprintStability",
    "Variation",
    "WindowSpectrum",
    "assess_stability",
    "normalise_residual",
    "spectral_distance",
    "window_spectrum",
]
