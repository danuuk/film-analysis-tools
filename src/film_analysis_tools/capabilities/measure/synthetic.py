"""Synthetic frame sequences with **known** grain amplitude and temporal correlation.

This exists so the residual extractor can be checked against an answer that is known in closed
form. The legacy grain path never had this, which is how a systematic bias in its amplitude
estimator survived: with only real footage to test on, there is no true value to compare against,
and a plausible number is indistinguishable from a correct one.

Grain is generated as a stationary AR(1) process across frames::

    g[0] ~ N(0, sigma^2)
    g[t] = rho * g[t-1] + sqrt(1 - rho^2) * eps[t],   eps ~ N(0, sigma^2)

so that ``Var(g[t]) = sigma^2`` exactly and ``Corr(g[t], g[t-k]) = rho^k``. Both parameters are
therefore recoverable, and an extractor that returns something else is wrong rather than merely
different.

Contaminants can be added deliberately — sub-pixel drift, fine texture, spatial correlation — to
check that the extractor either handles them or refuses the window, rather than quietly folding
them into the amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from film_analysis_tools.core.errors import SelectionError


@dataclass(frozen=True)
class SyntheticSpec:
    """Everything needed to reproduce a synthetic sequence, and its true parameters."""

    frames: int = 12
    height: int = 192
    width: int = 192
    sigma: float = 0.01
    """True per-frame grain standard deviation."""

    rho: float = 0.0
    """True lag-1 temporal correlation of the grain. ``rho**k`` at lag k."""

    base_level: float = 0.2
    texture_amplitude: float = 0.0
    """Amplitude of static scene detail. Fine texture is what leaks into a naive residual."""

    texture_scale_px: float = 6.0
    drift_px_per_frame: tuple[float, float] = (0.0, 0.0)
    """Sub-pixel or whole-pixel translation per frame, as (dy, dx)."""

    spatial_correlation_px: float = 0.0
    """Blur radius applied to each grain field. Non-zero means grain is not white."""

    seed: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -0.99 <= self.rho <= 0.99:
            raise SelectionError(f"rho must lie in [-0.99, 0.99]: {self.rho}")
        if self.sigma < 0.0:
            raise SelectionError(f"sigma must be non-negative: {self.sigma}")
        if self.frames < 2:
            raise SelectionError(f"need at least two frames: {self.frames}")

    def expected_difference_variance(self, lag: int) -> float:
        """``2 sigma^2 (1 - rho**lag)`` — the closed form an extractor must reproduce."""
        return 2.0 * self.sigma**2 * (1.0 - self.rho**lag)


def _static_texture(spec: SyntheticSpec) -> np.ndarray:
    """Deterministic band-limited detail, so a window can be given structure to align on."""
    if spec.texture_amplitude <= 0.0:
        return np.zeros((spec.height, spec.width), dtype=np.float64)
    ys = np.arange(spec.height, dtype=np.float64)[:, None]
    xs = np.arange(spec.width, dtype=np.float64)[None, :]
    period = max(spec.texture_scale_px, 1.0)
    pattern = (
        np.sin(2.0 * np.pi * xs / period)
        + np.sin(2.0 * np.pi * ys / (period * 1.37))
        + 0.5 * np.sin(2.0 * np.pi * (xs + ys) / (period * 2.11))
    )
    return spec.texture_amplitude * pattern / 2.5


def _shift_fourier(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Exact sub-pixel translation by Fourier phase ramp, for generating known drift."""
    if dy == 0.0 and dx == 0.0:
        return image
    height, width = image.shape
    v = np.fft.fftfreq(height)[:, None]
    u = np.fft.fftfreq(width)[None, :]
    ramp = np.exp(-2.0j * np.pi * (v * dy + u * dx))
    return np.real(np.fft.ifft2(np.fft.fft2(image) * ramp))


def _box_blur(image: np.ndarray, radius: float) -> np.ndarray:
    """Separable box blur, used to give grain a known spatial correlation."""
    size = max(1, round(radius) * 2 + 1)
    if size == 1:
        return image
    kernel = np.ones(size, dtype=np.float64) / size
    padded = np.pad(image, ((size // 2, size // 2), (0, 0)), mode="reflect")
    blurred = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 0, padded)
    padded = np.pad(blurred, ((0, 0), (size // 2, size // 2)), mode="reflect")
    return np.asarray(
        np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    )


def grain_fields(spec: SyntheticSpec) -> np.ndarray:
    """The AR(1) grain stack alone, shape ``(frames, height, width)``.

    Returned separately from the rendered sequence so a test can compare against the truth
    without the scene content in the way.
    """
    generator = np.random.default_rng(spec.seed)
    shape = (spec.height, spec.width)
    fields = np.empty((spec.frames, *shape), dtype=np.float64)

    innovation_scale = np.sqrt(max(1.0 - spec.rho**2, 0.0))
    fields[0] = generator.normal(0.0, spec.sigma, size=shape)
    for index in range(1, spec.frames):
        innovation = generator.normal(0.0, spec.sigma, size=shape)
        fields[index] = spec.rho * fields[index - 1] + innovation_scale * innovation

    if spec.spatial_correlation_px > 0.0:
        # Blurring reduces variance, so renormalise to keep sigma the declared truth.
        blurred = np.stack([_box_blur(field, spec.spatial_correlation_px) for field in fields])
        observed = float(np.std(blurred))
        if observed > 0.0:
            blurred *= spec.sigma / observed
        return blurred
    return fields


def sequence(spec: SyntheticSpec) -> np.ndarray:
    """A rendered frame sequence: base level, static texture, drift, plus the grain stack."""
    texture = _static_texture(spec)
    fields = grain_fields(spec)
    drift_y, drift_x = spec.drift_px_per_frame

    frames = np.empty_like(fields)
    for index in range(spec.frames):
        scene = spec.base_level + texture
        if drift_y or drift_x:
            scene = spec.base_level + _shift_fourier(texture, drift_y * index, drift_x * index)
        frames[index] = scene + fields[index]
    return frames


__all__ = ["SyntheticSpec", "grain_fields", "sequence"]
