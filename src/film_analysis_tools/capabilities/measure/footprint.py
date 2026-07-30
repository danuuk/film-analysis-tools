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
from dataclasses import dataclass, field
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


def _mean_power(residual: np.ndarray) -> np.ndarray:
    """Mean 2D power spectrum over a residual stack (DC included)."""
    power: np.ndarray | None = None
    for frame in residual:
        centred = frame - frame.mean()
        spectrum = np.abs(np.fft.fft2(centred)) ** 2
        power = spectrum if power is None else power + spectrum
    assert power is not None
    return power / residual.shape[0]


def _interp_crossing(line: np.ndarray, threshold: float = 0.5) -> float:
    """The lag at which ``line`` first falls below ``threshold``, linearly interpolated.

    Integer lags gave the coarsely quantised half-widths (an exact 2.0 anisotropy). Interpolating
    between the two straddling lags recovers a continuous width, which is what the anisotropy and
    directional comparisons need to mean anything finer than 2:1.
    """
    below = np.where(line < threshold)[0]
    if below.size == 0:
        return float(line.size)
    first = int(below[0])
    if first == 0:
        return 0.5
    high, low = line[first - 1], line[first]
    frac = (high - threshold) / max(high - low, EPS)
    return float(first - 1 + frac)


def _axis_slice(power: np.ndarray, axis: int) -> np.ndarray:
    """1D power along one frequency axis (the other frequency at 0), positive frequencies only."""
    line = power[0, :] if axis == 1 else power[:, 0]
    return np.asarray(line[: line.size // 2])


def _resample(profile: np.ndarray, bins: int) -> np.ndarray:
    """Resample a directional slice onto the same ``bins`` frequency grid as the radial profile."""
    source = np.linspace(0.0, 0.5, profile.size, endpoint=False)
    edges = np.linspace(0.0, 0.5, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(centres, source, profile)


def _directional_widths(residual: np.ndarray) -> tuple[float, float]:
    """Interpolated horizontal and vertical autocorrelation half-widths, in pixels."""
    power = _mean_power(residual)
    autocorr = np.fft.ifft2(power).real
    autocorr = autocorr / max(autocorr[0, 0], EPS)
    horizontal = _interp_crossing(autocorr[0, : autocorr.shape[1] // 2])
    vertical = _interp_crossing(autocorr[: autocorr.shape[0] // 2, 0])
    return horizontal, vertical


def _block_axis_peak(profile: np.ndarray) -> float:
    """Largest excess at a block fundamental or its low harmonics over the local background.

    Evaluated on a *directional* slice, where a codec grid concentrates its energy -- radial
    averaging dilutes exactly these narrow axial peaks. Harmonics are checked because a square
    grid rings at multiples of its fundamental.
    """
    freqs = np.linspace(0.0, 0.5, profile.size, endpoint=False)
    peak = 1.0
    for size in BLOCK_SIZES:
        for harmonic in (1, 2, 3):
            target = harmonic / size
            if target >= 0.5:
                continue
            index = int(np.argmin(np.abs(freqs - target)))
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
    horizontal_psd: tuple[float, ...]
    vertical_psd: tuple[float, ...]
    grain_radius: float
    grain_radius_h: float
    grain_radius_v: float
    anisotropy: float
    block_peak_h: float
    block_peak_v: float
    psd_2d: np.ndarray | None = field(default=None, repr=False, compare=False)
    """Full normalised 2D power spectrum, kept in memory for kernel materialisation and never
    serialised. ``None`` on a spectrum rebuilt from a record, which cannot re-form a kernel."""

    @property
    def block_peak(self) -> float:
        return max(self.block_peak_h, self.block_peak_v)

    def profile(self, direction: str) -> tuple[float, ...]:
        return {
            "radial": self.radial_psd,
            "horizontal": self.horizontal_psd,
            "vertical": self.vertical_psd,
        }[direction]

    def as_record(self) -> dict[str, Any]:
        return {
            "interval": self.interval,
            "level": self.level,
            "band": self.band,
            "grain_radius": self.grain_radius,
            "grain_radius_h": self.grain_radius_h,
            "grain_radius_v": self.grain_radius_v,
            "anisotropy": self.anisotropy,
            "block_peak_h": self.block_peak_h,
            "block_peak_v": self.block_peak_v,
            "radial_psd": list(self.radial_psd),
            "horizontal_psd": list(self.horizontal_psd),
            "vertical_psd": list(self.vertical_psd),
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
    centres, radial = _radial_psd(residual, bins=bins)
    power = _mean_power(residual)
    horizontal = _normalise_profile(centres, _resample(_axis_slice(power, axis=1), bins))
    vertical = _normalise_profile(centres, _resample(_axis_slice(power, axis=0), bins))
    width_h, width_v = _directional_widths(residual)
    return WindowSpectrum(
        interval=interval,
        level=level,
        band=band,
        frequencies=tuple(float(x) for x in centres),
        radial_psd=tuple(float(x) for x in _normalise_profile(centres, radial)),
        horizontal_psd=tuple(float(x) for x in horizontal),
        vertical_psd=tuple(float(x) for x in vertical),
        grain_radius=float(np.hypot(width_h, width_v) / np.sqrt(2.0)),
        grain_radius_h=width_h,
        grain_radius_v=width_v,
        anisotropy=width_h / max(width_v, EPS),
        block_peak_h=_block_axis_peak(np.asarray(horizontal)),
        block_peak_v=_block_axis_peak(np.asarray(vertical)),
        psd_2d=power / max(float(power.mean()), EPS),
    )


def spectral_distance(a: WindowSpectrum, b: WindowSpectrum, *, direction: str = "radial") -> float:
    """Log-spectral distance between two normalised profiles over the comparison band.

    ``direction`` selects the radial, horizontal or vertical profile. Radial alone discards
    orientation, so a level-dependent *anisotropy* is invisible to it; the directional profiles
    are what expose an orientation trend.
    """
    centres = np.asarray(a.frequencies)
    band = _in_band(centres)
    first = np.log(np.maximum(np.asarray(a.profile(direction))[band], EPS))
    second = np.log(np.maximum(np.asarray(b.profile(direction))[band], EPS))
    finite = np.isfinite(first) & np.isfinite(second)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean((first[finite] - second[finite]) ** 2)))


def _mean_spectrum(spectra: Sequence[WindowSpectrum]) -> WindowSpectrum:
    """The mean normalised profile over a group, kept as a WindowSpectrum for distance reuse."""
    first = spectra[0]
    return WindowSpectrum(
        interval="mean",
        level=float(np.mean([s.level for s in spectra])),
        band=first.band,
        frequencies=first.frequencies,
        radial_psd=tuple(float(x) for x in np.nanmean([s.radial_psd for s in spectra], axis=0)),
        horizontal_psd=tuple(
            float(x) for x in np.nanmean([s.horizontal_psd for s in spectra], axis=0)
        ),
        vertical_psd=tuple(float(x) for x in np.nanmean([s.vertical_psd for s in spectra], axis=0)),
        grain_radius=float(np.mean([s.grain_radius for s in spectra])),
        grain_radius_h=float(np.mean([s.grain_radius_h for s in spectra])),
        grain_radius_v=float(np.mean([s.grain_radius_v for s in spectra])),
        anisotropy=float(np.mean([s.anisotropy for s in spectra])),
        block_peak_h=float(np.mean([s.block_peak_h for s in spectra])),
        block_peak_v=float(np.mean([s.block_peak_v for s in spectra])),
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
    grain_radius_h: float
    grain_radius_v: float
    anisotropy: float
    block_peak_h: float
    block_peak_v: float
    level_min: float
    level_max: float

    def as_record(self) -> dict[str, Any]:
        return {
            "band": self.band,
            "windows": self.windows,
            "intervals": self.intervals,
            "grain_radius_h": self.grain_radius_h,
            "grain_radius_v": self.grain_radius_v,
            "anisotropy": self.anisotropy,
            "block_peak_h": self.block_peak_h,
            "block_peak_v": self.block_peak_v,
            "level_min": self.level_min,
            "level_max": self.level_max,
        }

    def line(self) -> str:
        return (
            f"    {self.band:9s} {self.windows:3d}w/{self.intervals}iv  "
            f"radius h/v {self.grain_radius_h:.2f}/{self.grain_radius_v:.2f}px  "
            f"anisotropy {self.anisotropy:.2f}  block h/v "
            f"{self.block_peak_h:.2f}/{self.block_peak_v:.2f}  "
            f"level {self.level_min:.4f}..{self.level_max:.4f}"
        )


def _band_summary(spectra: Sequence[WindowSpectrum]) -> BandSummary:
    return BandSummary(
        band=spectra[0].band,
        windows=len(spectra),
        intervals=len({s.interval for s in spectra}),
        grain_radius_h=float(np.median([s.grain_radius_h for s in spectra])),
        grain_radius_v=float(np.median([s.grain_radius_v for s in spectra])),
        anisotropy=float(np.median([s.anisotropy for s in spectra])),
        block_peak_h=float(np.median([s.block_peak_h for s in spectra])),
        block_peak_v=float(np.median([s.block_peak_v for s in spectra])),
        level_min=float(min(s.level for s in spectra)),
        level_max=float(max(s.level for s in spectra)),
    )


DIRECTIONS: tuple[str, ...] = ("radial", "horizontal", "vertical")


@dataclass(frozen=True)
class DirectionalStability:
    """Split-half, between-interval and between-level variation for one profile direction."""

    direction: str
    split_half: Variation
    between_interval: Variation
    between_level: Variation

    @property
    def one_footprint_suffices(self) -> bool | None:
        if not self.between_level.distances:
            return None
        baseline = max(self.split_half.p90, self.between_interval.p90)
        return self.between_level.median <= baseline

    def as_record(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "split_half": self.split_half.as_record(),
            "between_interval": self.between_interval.as_record(),
            "between_level": self.between_level.as_record(),
            "one_footprint_suffices": self.one_footprint_suffices,
        }

    def line(self) -> str:
        answer = self.one_footprint_suffices
        tag = {True: "stable", False: "LEVEL-DEPENDENT", None: "n/a"}[answer]
        return (
            f"  {self.direction:11s} split-half {self.split_half.median:.3f}  "
            f"between-iv {self.between_interval.median:.3f}  "
            f"between-level {self.between_level.median:.3f}  -> {tag}"
        )


@dataclass(frozen=True)
class FootprintStability:
    """Whether one common footprint suffices, judged radially *and* directionally.

    Radial averaging discards orientation, so it cannot see a footprint whose *anisotropy* changes
    with level while its radial shape stays constant. The horizontal and vertical profiles are
    compared against the same nulls, and one common footprint is adopted only when **every**
    direction is stable.
    """

    directions: tuple[DirectionalStability, ...]
    bands_present: tuple[str, ...]
    bands: tuple[BandSummary, ...] = ()

    def direction(self, name: str) -> DirectionalStability:
        for one in self.directions:
            if one.direction == name:
                return one
        raise DataError(f"no directional stability for {name!r}")

    @property
    def one_footprint_suffices(self) -> bool | None:
        if len(self.bands_present) < 2:
            return None
        answers = [one.one_footprint_suffices for one in self.directions]
        if any(answer is False for answer in answers):
            return False
        if all(answer is True for answer in answers):
            return True
        return None

    def as_record(self) -> dict[str, Any]:
        return {
            "bands_present": list(self.bands_present),
            "bands": [band.as_record() for band in self.bands],
            "directions": [one.as_record() for one in self.directions],
            "one_footprint_suffices": self.one_footprint_suffices,
        }

    def summary(self) -> str:
        verdict = {True: "ONE footprint suffices", False: "level-dependent footprint indicated"}
        lines = ["spatial footprint stability (log-spectral distance, median):"]
        lines += [one.line() for one in self.directions]
        for band in self.bands:
            lines.append(band.line())
        answer = self.one_footprint_suffices
        if answer is None:
            lines.append("  verdict: cannot assess -- fewer than two level bands present")
        else:
            lines.append(f"  verdict: {verdict[answer]}")
        return "\n".join(lines)


def _split_half_distances(
    spectra: Sequence[WindowSpectrum], *, direction: str, seed: int = 0
) -> list[float]:
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
        distances.append(
            spectral_distance(_mean_spectrum(left), _mean_spectrum(right), direction=direction)
        )
    return distances


def _between_group_distances(
    spectra: Sequence[WindowSpectrum], *, outer: str, inner: str, direction: str
) -> list[float]:
    """Distances between mean spectra of different ``inner`` groups, within each ``outer`` group.

    ``outer='band', inner='interval'`` gives between-interval variation; swapping them gives
    between-level.
    """
    nested: dict[str, dict[str, list[WindowSpectrum]]] = {}
    for spectrum in spectra:
        okey = getattr(spectrum, outer)
        ikey = getattr(spectrum, inner)
        nested.setdefault(okey, {}).setdefault(ikey, []).append(spectrum)
    distances: list[float] = []
    for groups in nested.values():
        means = {
            key: _mean_spectrum(members) for key, members in groups.items() if len(members) >= 2
        }
        keys = sorted(means)
        for left in range(len(keys)):
            for right in range(left + 1, len(keys)):
                distances.append(
                    spectral_distance(means[keys[left]], means[keys[right]], direction=direction)
                )
    return distances


def _directional_stability(
    spectra: Sequence[WindowSpectrum], *, direction: str, seed: int
) -> DirectionalStability:
    return DirectionalStability(
        direction=direction,
        split_half=Variation(
            "split_half", tuple(_split_half_distances(spectra, direction=direction, seed=seed))
        ),
        between_interval=Variation(
            "between_interval",
            tuple(
                _between_group_distances(
                    spectra, outer="band", inner="interval", direction=direction
                )
            ),
        ),
        between_level=Variation(
            "between_level",
            tuple(
                _between_group_distances(
                    spectra, outer="interval", inner="band", direction=direction
                )
            ),
        ),
    )


def assess_stability(spectra: Sequence[WindowSpectrum], *, seed: int = 0) -> FootprintStability:
    """Run the split-half, between-interval and between-level comparisons in every direction."""
    if len(spectra) < 4:
        raise DataError(f"need at least 4 window spectra to assess stability; got {len(spectra)}")
    by_band: dict[str, list[WindowSpectrum]] = {}
    for spectrum in spectra:
        by_band.setdefault(spectrum.band, []).append(spectrum)
    return FootprintStability(
        directions=tuple(
            _directional_stability(spectra, direction=name, seed=seed) for name in DIRECTIONS
        ),
        bands_present=tuple(sorted(by_band)),
        bands=tuple(_band_summary(members) for _, members in sorted(by_band.items())),
    )


__all__ = [
    "BLOCK_SIZES",
    "COMPARE_BAND",
    "DIRECTIONS",
    "BandSummary",
    "DirectionalStability",
    "FootprintStability",
    "Variation",
    "WindowSpectrum",
    "assess_stability",
    "normalise_residual",
    "spectral_distance",
    "window_spectrum",
]
