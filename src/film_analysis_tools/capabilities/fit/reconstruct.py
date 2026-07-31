"""Held-out reconstruction: materialise the candidate grain and test it against unseen material.

The amplitude fit and the footprint stability each answered one question. Neither *renders* grain:
the three summary numbers -- sigma(L), a radius, an anisotropy -- are satisfied by many different
filters. This slice materialises the actual candidate and checks whether it reproduces the measured
statistics of an interval it was **not** built from.

The discipline is leave-one-interval-out, applied to *both* halves of the model at once:

* the amplitude law is fitted from the other intervals;
* the 2D power spectrum is pooled from the other intervals, with **equal weight per interval** so a
  window-rich scene cannot dominate the kernel;
* a real-valued filter is taken as ``sqrt(PSD)``, driven with independent Gaussian fields, and
  scaled by the held-out amplitude prediction;
* every comparison is made against the held interval alone.

So the held interval influences neither the amplitude nor the spatial structure of what is compared
to it. The midtone candidate is Gaussian by construction; the shadow heavy tails are deliberately
excluded, since they are a delivery artefact rather than the particle law (see the footprint and
tail studies).

**What this validates, precisely.** The generator is linear -- a unit-variance shaped field times
a per-pixel amplitude -- so its parts are validated separately and honestly: the *spatial* structure
and *temporal* independence on the unit-variance field, and the *amplitude* law against the
held-out points. This is a held-out validation of all three components, not yet a held-out
reconstruction of a spatially-varying image through the full amplitude-times-footprint generator;
that composed check belongs in the A/B renderer, which multiplies the two together over a real
level field.

The narrow codec block peaks are **not** reproduced: :func:`_suppress_codec` interpolates across the
block-frequency bins when materialising the kernel, on the judgement that they are a
delivery/mastering artefact. The measured peaks are retained as provenance on each fold so the
exclusion is explicit rather than accidental.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from film_analysis_tools.capabilities.fit.amplitude import AmplitudePoint, fit_power_floor
from film_analysis_tools.capabilities.measure import footprint as fp
from film_analysis_tools.core.errors import DataError

EPS = 1.0e-12


def _suppress_codec(pooled: np.ndarray) -> np.ndarray:
    """Interpolate across the axial codec-block frequency bins, leaving the broad footprint.

    The narrow axial peaks at 1/8 and 1/16 (and harmonics) are a delivery/mastering artefact --
    the encoder's block grid, not the grain's particle law. Reproducing them would print a
    periodic pattern that only gets worse when the plugin output is re-encoded, so they are
    replaced by a smooth interpolation from neighbouring frequency bins on the same axis. The peaks
    are kept as *provenance* (measured block_peak on the WindowSpectrum), not as generator
    behaviour.
    """
    result = pooled.astype(np.float64).copy()
    n = result.shape[0]
    targets: set[int] = set()
    for size in fp.BLOCK_SIZES:
        for harmonic in (1, 2, 3):
            bin_index = round(harmonic / size * n)
            if 2 < bin_index < n - 2:
                targets.update({bin_index, n - bin_index})
    for k in sorted(targets):
        for lo, hi in ((k - 2, k + 2),):
            if lo >= 0 and hi < n:
                result[0, k] = 0.5 * (result[0, lo] + result[0, hi])  # horizontal axis
                result[k, 0] = 0.5 * (result[lo, 0] + result[hi, 0])  # vertical axis
    return result


def materialise_filter(
    spectra: Sequence[fp.WindowSpectrum], *, suppress_codec: bool = True
) -> np.ndarray:
    """A real-valued frequency filter from the interval-weighted, symmetrised pooled 2D PSD.

    Each interval contributes equally, not each window. The pooled power is symmetrised so the
    implied field is exactly real, and the filter magnitude is ``sqrt(power)``. ``suppress_codec``
    (default true) interpolates across the block-frequency axial bins so the grain generator does
    not reproduce the codec grid -- see :func:`_suppress_codec`.
    """
    usable = [s for s in spectra if s.psd_2d is not None]
    if not usable:
        raise DataError("cannot materialise a filter: no window carries a 2D PSD")
    shape = usable[0].psd_2d.shape  # type: ignore[union-attr]
    if any(s.psd_2d.shape != shape for s in usable):  # type: ignore[union-attr]
        raise DataError("pooling needs a common tile size; got mixed 2D PSD shapes")

    counts: dict[str, int] = {}
    for spectrum in usable:
        counts[spectrum.interval] = counts.get(spectrum.interval, 0) + 1

    pooled = np.zeros(shape, dtype=np.float64)
    total = 0.0
    for spectrum in usable:
        weight = 1.0 / counts[spectrum.interval]
        pooled += weight * np.asarray(spectrum.psd_2d)
        total += weight
    pooled /= max(total, EPS)
    # Enforce exact even symmetry so sqrt(pooled) is an even filter and ifft2 yields a real field.
    # The conjugate partner of an *unshifted* FFT bin k is (-k) mod N, i.e. a reversal followed by
    # a one-bin roll -- not the bare reversal ``pooled[::-1, ::-1]``, which maps k to N-1-k and
    # broke Hermitian symmetry so badly that .real silently discarded ~17% of the field. A pooled
    # |FFT|^2 average is already Hermitian, so this is a no-op up to float noise; it stays as a
    # guard, now correct.
    mirror = np.roll(pooled[::-1, ::-1], shift=(1, 1), axis=(0, 1))
    pooled = 0.5 * (pooled + mirror)
    if suppress_codec:
        pooled = _suppress_codec(pooled)
    return np.sqrt(np.maximum(pooled, 0.0))


def generate_frames(filter_magnitude: np.ndarray, frames: int, *, seed: int = 0) -> np.ndarray:
    """Independent Gaussian fields shaped by the filter, each at unit variance.

    Independent per frame, so their temporal correlation is zero by construction -- the runtime
    baseline the temporal study established for this source.
    """
    rng = np.random.default_rng(seed)
    height, width = filter_magnitude.shape
    stack = np.empty((frames, height, width), dtype=np.float64)
    for index in range(frames):
        white = rng.normal(0.0, 1.0, (height, width))
        shaped = np.fft.ifft2(np.fft.fft2(white) * filter_magnitude).real
        std = float(shaped.std())
        stack[index] = shaped / std if std > EPS else shaped
    return stack


def materialise_kernel(filter_magnitude: np.ndarray, *, support: int = 15) -> np.ndarray:
    """A small, unit-L2 spatial kernel from the frequency filter -- the footprint's runtime form.

    Convolving white noise (unit variance) with a unit-L2 kernel gives a unit-variance field with
    the measured footprint, at any frame size. Cropping to a compact support is honest here because
    the footprint is ~1 px: essentially all of its energy sits within a few pixels of the centre.
    """
    kernel = np.fft.fftshift(np.fft.ifft2(filter_magnitude).real)
    centre = filter_magnitude.shape[0] // 2
    half = support // 2
    cropped = kernel[centre - half : centre + half + 1, centre - half : centre + half + 1]
    norm = float(np.sqrt(np.sum(cropped**2)))
    return cropped / norm if norm > EPS else cropped


@dataclass(frozen=True)
class SupportCheck:
    """How faithfully a compact kernel of a given support reproduces the full filter."""

    support: int
    retained_energy: float
    radial_psd_distance: float
    horizontal_psd_distance: float
    vertical_psd_distance: float
    grain_radius_h: float
    grain_radius_v: float
    anisotropy: float

    def as_record(self) -> dict[str, Any]:
        return {
            "support": self.support,
            "retained_energy": self.retained_energy,
            "radial_psd_distance": self.radial_psd_distance,
            "horizontal_psd_distance": self.horizontal_psd_distance,
            "vertical_psd_distance": self.vertical_psd_distance,
            "grain_radius_h": self.grain_radius_h,
            "grain_radius_v": self.grain_radius_v,
            "anisotropy": self.anisotropy,
        }

    def line(self) -> str:
        return (
            f"    support {self.support:2d}: retained energy {self.retained_energy:6.2%}  "
            f"radial {self.radial_psd_distance:.3f}  h/v {self.horizontal_psd_distance:.3f}/"
            f"{self.vertical_psd_distance:.3f}  radius h/v {self.grain_radius_h:.2f}/"
            f"{self.grain_radius_v:.2f}  aniso {self.anisotropy:.2f}"
        )


def kernel_support_report(
    spectra: Sequence[fp.WindowSpectrum],
    *,
    supports: Sequence[int] = (15, 21, 31),
    frames: int = 60,
    seed: int = 0,
) -> tuple[SupportCheck, ...]:
    """Fields from the full filter versus its compact kernel, across support sizes.

    The compact kernel is what ``render_candidate`` actually uses at runtime; this quantifies what
    cropping the full footprint to a small window costs, so the smallest support that preserves the
    measured character can be chosen rather than assumed.
    """
    filter_magnitude = materialise_filter(spectra)
    full_kernel = np.fft.fftshift(np.fft.ifft2(filter_magnitude).real)
    total_energy = float(np.sum(full_kernel**2))
    reference = fp.window_spectrum(
        generate_frames(filter_magnitude, frames, seed=seed),
        interval="full",
        level=0.1,
        band="m",
        sigma_hat=1.0,
    )

    checks: list[SupportCheck] = []
    for support in supports:
        centre = filter_magnitude.shape[0] // 2
        half = support // 2
        cropped = full_kernel[centre - half : centre + half + 1, centre - half : centre + half + 1]
        retained = float(np.sum(cropped**2)) / max(total_energy, EPS)

        kernel = materialise_kernel(filter_magnitude, support=support)
        level = np.full((160, 160), 0.1)
        rendered = render_candidate(level, _ConstAmp(1.0), kernel, frames=frames, seed=seed + 1)
        gen = fp.window_spectrum(rendered, interval="k", level=0.1, band="m", sigma_hat=1.0)
        checks.append(
            SupportCheck(
                support=support,
                retained_energy=retained,
                radial_psd_distance=fp.spectral_distance(gen, reference, direction="radial"),
                horizontal_psd_distance=fp.spectral_distance(
                    gen, reference, direction="horizontal"
                ),
                vertical_psd_distance=fp.spectral_distance(gen, reference, direction="vertical"),
                grain_radius_h=gen.grain_radius_h,
                grain_radius_v=gen.grain_radius_v,
                anisotropy=gen.anisotropy,
            )
        )
    return tuple(checks)


class _ConstAmp:
    """A constant amplitude model, for exercising the renderer at unit strength."""

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def predict(self, level: np.ndarray, *, outside: str = "clamp") -> np.ndarray:
        return np.full_like(np.asarray(level, dtype=np.float64), self.sigma)


def render_candidate(
    level: np.ndarray,
    amplitude_model: Any,
    kernel: np.ndarray,
    *,
    frames: int,
    seed: int = 0,
) -> np.ndarray:
    """The **composed** generator: level image + footprint field x sigma(level), per frame.

    This is the whole candidate applied to a spatially-varying image -- the composition the
    componentwise held-out validation does not itself perform. ``level`` is a linear-luma image;
    each frame adds an independent footprint field scaled per pixel by the fitted amplitude law.

    **Runtime-faithful.** The field is white unit Gaussian noise convolved with the unit-L2 kernel,
    with *no* frame-global normalisation. A real-time shader convolves noise with a fixed kernel
    and cannot divide by a per-frame standard deviation; a unit-L2 kernel already yields
    unit-variance output in expectation, so the aggregate RMS follows ``sigma(L)`` on its own,
    within ordinary sampling uncertainty. Dividing by the realised per-frame std would flatter the
    result by removing exactly the sampling variation the runtime has.
    """
    sigma = np.asarray(amplitude_model.predict(level, outside="clamp"), dtype=np.float64)
    rng = np.random.default_rng(seed)
    height, width = level.shape
    padded = np.zeros((height, width))
    kh, kw = kernel.shape
    padded[:kh, :kw] = kernel
    padded = np.roll(padded, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    transfer = np.fft.fft2(padded)

    out = np.empty((frames, height, width), dtype=np.float64)
    for index in range(frames):
        white = rng.normal(0.0, 1.0, (height, width))
        field = np.fft.ifft2(np.fft.fft2(white) * transfer).real
        out[index] = level + sigma * field
    return out


def luma_delta_to_rgb(luma_delta: np.ndarray, weights: tuple[float, float, float]) -> np.ndarray:
    """Project a scalar luma grain delta into RGB the way the legacy shader injects it.

    ``dRGB = w * (dL / dot(w, w))`` -- the minimum-norm RGB change whose luma is ``dL``. This is
    the shader's ``injectionDelta``; equal-RGB injection would add a colour shift unrelated to the
    footprint, confounding grain character with a tint in any A/B. Default weights are Rec.709.
    """
    w = np.asarray(weights, dtype=np.float64)
    scale = luma_delta / max(float(np.dot(w, w)), EPS)
    return np.asarray(scale[..., None] * w)


REC709_LUMA: tuple[float, float, float] = (0.2126, 0.7152, 0.0722)


@dataclass(frozen=True)
class FoldResult:
    """One leave-one-interval-out fold: the held interval, and how well the candidate matched it."""

    held_interval: str
    train_intervals: int
    amplitude_log_error: float
    radial_psd_distance: float
    horizontal_psd_distance: float
    vertical_psd_distance: float
    anisotropy_measured: float
    anisotropy_generated: float
    grain_radius_h_error: float
    grain_radius_v_error: float
    block_peak_measured: float
    block_peak_generated: float
    generated_skew: float
    generated_excess_kurtosis: float
    generated_rho: float

    def as_record(self) -> dict[str, Any]:
        return {
            "held_interval": self.held_interval,
            "train_intervals": self.train_intervals,
            "amplitude_log_error": self.amplitude_log_error,
            "radial_psd_distance": self.radial_psd_distance,
            "horizontal_psd_distance": self.horizontal_psd_distance,
            "vertical_psd_distance": self.vertical_psd_distance,
            "anisotropy_measured": self.anisotropy_measured,
            "anisotropy_generated": self.anisotropy_generated,
            "grain_radius_h_error": self.grain_radius_h_error,
            "grain_radius_v_error": self.grain_radius_v_error,
            "block_peak_measured": self.block_peak_measured,
            "block_peak_generated": self.block_peak_generated,
            "generated_skew": self.generated_skew,
            "generated_excess_kurtosis": self.generated_excess_kurtosis,
            "generated_rho": self.generated_rho,
        }


@dataclass(frozen=True)
class Reconstruction:
    """The held-out reconstruction across all folds, and its aggregate errors."""

    folds: tuple[FoldResult, ...]
    frames: int
    tile: int

    def _spread(self, pick: Any) -> tuple[float, float]:
        values = [pick(one) for one in self.folds]
        return (float(np.median(values)), float(np.max(values))) if values else (0.0, 0.0)

    def as_record(self) -> dict[str, Any]:
        radial = self._spread(lambda f: f.radial_psd_distance)
        amp = self._spread(lambda f: f.amplitude_log_error)
        return {
            "validation": "leave-one-interval-out reconstruction",
            "folds": len(self.folds),
            "frames_per_fold": self.frames,
            "tile": self.tile,
            "amplitude_log_error_median": amp[0],
            "radial_psd_distance_median": radial[0],
            "fold_results": [one.as_record() for one in self.folds],
        }

    def summary(self) -> str:
        amp_med, amp_max = self._spread(lambda f: f.amplitude_log_error)
        rad_med, rad_max = self._spread(lambda f: f.radial_psd_distance)
        hor_med, _ = self._spread(lambda f: f.horizontal_psd_distance)
        ver_med, _ = self._spread(lambda f: f.vertical_psd_distance)
        skew_med, _ = self._spread(lambda f: abs(f.generated_skew))
        kurt_med, _ = self._spread(lambda f: abs(f.generated_excess_kurtosis))
        rho_med, _ = self._spread(lambda f: abs(f.generated_rho))
        lines = [
            f"held-out reconstruction ({len(self.folds)} folds, {self.frames} frames, "
            f"{self.tile}px tile):",
            f"  amplitude    log-error median {amp_med:.3f}  max {amp_max:.3f}",
            f"  radial PSD   distance median {rad_med:.3f}  max {rad_max:.3f}",
            f"  directional  horizontal {hor_med:.3f}  vertical {ver_med:.3f}  (median distance)",
            "  per fold (held interval): "
            + "; ".join(
                f"{f.held_interval}s aniso "
                f"{f.anisotropy_measured:.2f}->{f.anisotropy_generated:.2f}"
                for f in self.folds
            ),
            f"  generated field is Gaussian and independent: |skew| {skew_med:.2f}  "
            f"|excess kurtosis| {kurt_med:.2f}  |rho| {rho_med:.3f}",
            f"  codec block peak: measured {self._spread(lambda f: f.block_peak_measured)[0]:.2f} "
            f"vs generated {self._spread(lambda f: f.block_peak_generated)[0]:.2f} "
            "(deliberately suppressed -- delivery artefact, not grain)",
        ]
        return "\n".join(lines)


def _skew_kurtosis(values: np.ndarray) -> tuple[float, float]:
    centred = values - values.mean()
    scale = float(np.std(centred))
    if scale <= EPS:
        return (0.0, 0.0)
    normalised = centred / scale
    return (float(np.mean(normalised**3)), float(np.mean(normalised**4) - 3.0))


def _generated_spectrum(frames: np.ndarray, band: str) -> fp.WindowSpectrum:
    return fp.window_spectrum(frames, interval="generated", level=0.1, band=band, sigma_hat=1.0)


def reconstruct(
    spectra: Sequence[fp.WindowSpectrum],
    points: Sequence[AmplitudePoint],
    *,
    frames: int = 24,
    seed: int = 0,
) -> Reconstruction:
    """Leave-one-interval-out reconstruction over every interval with enough evidence.

    For each held interval: the amplitude law and the pooled kernel come from the *other*
    intervals only, a sequence is generated and re-measured, and it is compared against the held
    interval's own measured spectra.
    """
    usable = [s for s in spectra if s.psd_2d is not None]
    intervals = sorted({s.interval for s in usable})
    if len(intervals) < 2:
        raise DataError("held-out reconstruction needs at least two intervals with 2D PSDs")

    tile = int(usable[0].psd_2d.shape[0])  # type: ignore[union-attr]
    results: list[FoldResult] = []
    for held in intervals:
        train_spectra = [s for s in usable if s.interval != held]
        held_spectra = [s for s in usable if s.interval == held]
        train_points = [p for p in points if p.interval != held]
        held_points = [p for p in points if p.interval == held]
        if not train_spectra or not held_spectra or len(set(p.interval for p in train_points)) < 2:
            continue

        amplitude_model = fit_power_floor(train_points)
        filter_magnitude = materialise_filter(train_spectra)
        generated = generate_frames(filter_magnitude, frames, seed=seed)
        band = max(
            {s.band for s in held_spectra},
            key=lambda b: sum(1 for s in held_spectra if s.band == b),
        )
        gen = _generated_spectrum(generated, band)

        held_mean = fp._mean_spectrum(held_spectra)
        residual = np.diff(generated, axis=0)
        skew, kurtosis = _skew_kurtosis(residual.ravel())
        estimate_rho = _lag_ratio_rho(generated)

        amp_error = 0.0
        if held_points:
            predicted = amplitude_model.predict(
                np.array([p.level for p in held_points]), outside="clamp"
            )
            observed = np.array([p.sigma for p in held_points])
            amp_error = float(
                np.median(
                    np.abs(np.log(np.maximum(predicted, EPS)) - np.log(np.maximum(observed, EPS)))
                )
            )

        results.append(
            FoldResult(
                held_interval=held,
                train_intervals=len({p.interval for p in train_points}),
                amplitude_log_error=amp_error,
                radial_psd_distance=fp.spectral_distance(gen, held_mean, direction="radial"),
                horizontal_psd_distance=fp.spectral_distance(
                    gen, held_mean, direction="horizontal"
                ),
                vertical_psd_distance=fp.spectral_distance(gen, held_mean, direction="vertical"),
                anisotropy_measured=held_mean.anisotropy,
                anisotropy_generated=gen.anisotropy,
                grain_radius_h_error=abs(gen.grain_radius_h - held_mean.grain_radius_h),
                grain_radius_v_error=abs(gen.grain_radius_v - held_mean.grain_radius_v),
                block_peak_measured=held_mean.block_peak,
                block_peak_generated=gen.block_peak,
                generated_skew=skew,
                generated_excess_kurtosis=kurtosis,
                generated_rho=estimate_rho,
            )
        )
    if not results:
        raise DataError("no fold had two training intervals and a held interval")
    return Reconstruction(folds=tuple(results), frames=frames, tile=tile)


def _lag_ratio_rho(frames: np.ndarray) -> float:
    """Lag-1 correlation of a generated stack, from the lag-2 to lag-1 difference-variance ratio."""
    if frames.shape[0] < 3:
        return 0.0
    var1 = float(np.var(frames[1:] - frames[:-1]))
    var2 = float(np.var(frames[2:] - frames[:-2]))
    return float(np.clip(var2 / var1 - 1.0, -1.0, 1.0)) if var1 > EPS else 0.0


__all__ = [
    "REC709_LUMA",
    "FoldResult",
    "Reconstruction",
    "SupportCheck",
    "generate_frames",
    "kernel_support_report",
    "luma_delta_to_rgb",
    "materialise_filter",
    "materialise_kernel",
    "reconstruct",
    "render_candidate",
]
