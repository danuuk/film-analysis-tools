"""Residual extraction: recovering grain amplitude and temporal correlation from a sequence.

Replaces the legacy estimator, which formed a raw pixel-wise temporal difference and then divided
its variance by two. Two things go wrong there, and they pull in opposite directions:

* dividing by two assumes consecutive frames are **uncorrelated**, which understates amplitude
  whenever grain persists across frames; and
* differencing without alignment lets sub-pixel motion move fine detail into the residual, which
  overstates amplitude on any window with texture in it. The legacy preset carries the evidence:
  one scene's static *textured* residual measured 7.15x its static grain-dominated residual.

Neither error was quantified, so they could not be distinguished from grain.

## Estimating correlation instead of assuming it

For a stationary process with lag-k correlation :math:`\\rho^k`,

.. math::  \\operatorname{Var}(g_t - g_{t-k}) = 2\\sigma^2 (1 - \\rho^k)

so the *ratio* of difference variances at lag 2 and lag 1 gives correlation directly:

.. math::  \\frac{\\operatorname{Var}(\\Delta_2)}{\\operatorname{Var}(\\Delta_1)}
           = \\frac{1-\\rho^2}{1-\\rho} = 1 + \\rho

Correlation is therefore estimated as ``ratio - 1``, and amplitude follows as
:math:`\\sigma^2 = \\operatorname{Var}(\\Delta_1) / (2(1-\\rho))`. Lag 4 provides an independent
check, since it must satisfy :math:`1 + \\rho + \\rho^2 + \\rho^3`.

This deliberately works from *differences* rather than from the undifferenced frames. Raw frames
are dominated by static scene content, whose autocorrelation is near 1 regardless of the grain, so
autocorrelating them measures the scene rather than the noise. Differencing removes the scene.

The legacy formula is the :math:`\\rho = 0` corner of this one, so agreement at zero correlation is
a property worth testing rather than a coincidence.

## Why alignment is integer-only

Resampling a frame by a fractional offset **changes the noise it carries** — interpolation reduces
variance and introduces spatial correlation. Sub-pixel alignment would therefore corrupt the exact
quantity being measured. So the shift is estimated to sub-pixel precision but applied only as
whole pixels, which is exact and leaves noise statistics untouched. The remaining fractional
offset is reported, so a window with persistent sub-pixel drift can be **rejected** rather than
silently measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from film_analysis_tools.core.errors import DataError

DEFAULT_LAGS: tuple[int, ...] = (1, 2, 4)
DEFAULT_BLUR_RADIUS = 3
DEFAULT_MOTION_BLUR_RADIUS = 4

#: Fractional offset above which a window is considered to be drifting rather than static.
SUBPIXEL_REJECT = 0.25

#: Largest shift considered. This is not a general motion tracker: windows are selected for being
#: nearly static, so motion beyond a few pixels means the window should be **rejected** rather than
#: tracked. Bounding the search also removes the failure mode where an unbounded peak search on a
#: featureless window locks onto noise and returns a 45-pixel shift.
DEFAULT_MAX_SHIFT = 4

#: Minimum fractional reduction in low-pass residual energy for a shift to be accepted. A spurious
#: shift is not harmless — displacing a frame by even one pixel destroys the frame-to-frame grain
#: correlation this module exists to measure — so a shift has to earn its application by
#: demonstrably reducing structured residual.
MIN_ALIGNMENT_GAIN = 0.05

#: Minimum ratio of low-pass scene structure to expected low-pass grain for a window to be
#: alignable at all. Measured across synthetic windows this sits at **exactly 1.0** when there is
#: no scene detail — independently of grain amplitude, which is what makes the ratio the right
#: quantity — and at 1.7 or above wherever there is structure worth aligning on.
#:
#: Without this gate, a featureless window with *negatively* correlated grain is actively
#: mis-aligned: anti-correlation makes zero lag the *worst* cross-correlation, so the peak search
#: moves away from it and the measured correlation collapses toward zero.
MIN_STRUCTURE_SNR = 1.5
EPS = 1.0e-12


# --------------------------------------------------------------------------- alignment


@dataclass(frozen=True)
class Shift:
    """Estimated translation between two frames, and whether it earned its application."""

    dy: int
    dx: int
    subpixel_dy: float
    subpixel_dx: float
    gain: float
    """Fractional reduction in low-pass residual energy this shift achieves."""

    structure: float
    structure_snr: float
    """Low-pass scene structure divided by expected low-pass grain. 1.0 means no structure."""

    at_bound: bool
    """True when the peak sat on the search boundary: motion exceeds what a static
    window allows."""

    @property
    def subpixel_magnitude(self) -> float:
        return float(np.hypot(self.subpixel_dy, self.subpixel_dx))

    @property
    def accepted(self) -> bool:
        """Whether applying this shift measurably improves the residual.

        The test is the shift's own effect, not a threshold on image content — which is what makes
        it robust on featureless windows, where any estimate is noise-driven and applying it would
        corrupt the correlation measurement for no benefit.
        """
        return (
            (self.dy != 0 or self.dx != 0)
            and self.structure_snr >= MIN_STRUCTURE_SNR
            and self.gain >= MIN_ALIGNMENT_GAIN
        )


def box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur. Public because window selection needs the same low-pass."""
    size = max(1, int(radius) * 2 + 1)
    if size == 1:
        return image
    kernel = np.ones(size, dtype=np.float64) / size
    padded = np.pad(image, ((size // 2, size // 2), (0, 0)), mode="reflect")
    rows = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 0, padded)
    padded = np.pad(rows, ((0, 0), (size // 2, size // 2)), mode="reflect")
    return np.asarray(
        np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    )


def _parabolic_offset(low: float, mid: float, high: float) -> float:
    """Sub-sample peak position from three samples, for reporting the fractional residue."""
    denominator = low - 2.0 * mid + high
    if abs(denominator) < EPS:
        return 0.0
    return float(np.clip(0.5 * (low - high) / denominator, -1.0, 1.0))


def estimate_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    blur_radius: int = DEFAULT_BLUR_RADIUS,
    max_shift: int = DEFAULT_MAX_SHIFT,
    motion_blur_radius: int = DEFAULT_MOTION_BLUR_RADIUS,
) -> Shift:
    """Translation that maps ``moving`` onto ``reference``, estimated on a low-pass copy.

    Low-passing first is what makes this a *motion* estimate rather than a noise estimate: grain
    is white and contributes nothing coherent to the correlation peak.

    Uses **plain cross-correlation**, not phase correlation. Fully whitening the cross-power
    spectrum weights noise-dominated high frequencies equally with real structure; measured against
    a known 2-pixel shift it returned 3 pixels *with higher apparent confidence* than the
    unnormalised correlation, which returned the right answer.
    """
    if reference.shape != moving.shape:
        raise DataError(f"shape mismatch: {reference.shape} vs {moving.shape}")

    a = box_blur(np.asarray(reference, dtype=np.float64), blur_radius)
    b = box_blur(np.asarray(moving, dtype=np.float64), blur_radius)
    a = a - a.mean()
    b = b - b.mean()
    structure = float(np.std(a))

    height, width = a.shape
    window = np.hanning(height)[:, None] * np.hanning(width)[None, :]
    surface = np.real(np.fft.ifft2(np.fft.fft2(a * window) * np.conj(np.fft.fft2(b * window))))

    # Search only within the bound: beyond it the window is not static and belongs rejected.
    reach = max(1, min(int(max_shift), min(height, width) // 2 - 1))
    offsets = np.arange(-reach, reach + 1)
    patch = surface[np.ix_(offsets % height, offsets % width)]
    local = int(np.argmax(patch))
    local_y, local_x = divmod(local, patch.shape[1])
    signed_y = int(offsets[local_y])
    signed_x = int(offsets[local_x])

    peak_y, peak_x = signed_y % height, signed_x % width
    offset_y = _parabolic_offset(
        float(surface[(peak_y - 1) % height, peak_x]),
        float(surface[peak_y, peak_x]),
        float(surface[(peak_y + 1) % height, peak_x]),
    )
    offset_x = _parabolic_offset(
        float(surface[peak_y, (peak_x - 1) % width]),
        float(surface[peak_y, peak_x]),
        float(surface[peak_y, (peak_x + 1) % width]),
    )

    # Does applying this shift actually reduce structured residual? A shift that cannot show a
    # gain is a noise-driven estimate and must not be applied.
    reference_raw = np.asarray(reference, dtype=np.float64)
    moving_raw = np.asarray(moving, dtype=np.float64)
    before = box_blur(moving_raw - reference_raw, motion_blur_radius)
    shifted = np.roll(moving_raw, (signed_y, signed_x), axis=(0, 1))
    after = box_blur(shifted - reference_raw, motion_blur_radius)
    margin = abs(signed_y) + abs(signed_x) + motion_blur_radius
    if margin * 2 < min(height, width):
        inner = (slice(margin, -margin), slice(margin, -margin))
        before, after = before[inner], after[inner]
    energy_before = float(np.mean(before**2))
    energy_after = float(np.mean(after**2))
    gain = 1.0 - energy_after / energy_before if energy_before > EPS else 0.0

    # Is there scene detail here at all, or only blurred grain? Estimate the grain level from the
    # high-pass of the *aligned* difference: using the raw difference would fold motion into the
    # grain estimate and depress the ratio on exactly the drifting windows that need aligning.
    aligned_delta = shifted - reference_raw
    high_pass_delta = aligned_delta - box_blur(aligned_delta, blur_radius)
    grain_level = float(np.std(high_pass_delta)) / np.sqrt(2.0)
    expected_low_pass_grain = grain_level / float(2 * max(blur_radius, 1) + 1)
    structure_snr = structure / max(expected_low_pass_grain, EPS)

    return Shift(
        dy=signed_y,
        dx=signed_x,
        subpixel_dy=offset_y,
        subpixel_dx=offset_x,
        gain=float(gain),
        structure=structure,
        structure_snr=float(structure_snr),
        at_bound=bool(abs(signed_y) >= reach or abs(signed_x) >= reach),
    )


def apply_shift(image: np.ndarray, shift: Shift) -> np.ndarray:
    """Whole-pixel translation only. Exact, and leaves the noise statistics untouched."""
    if shift.dy == 0 and shift.dx == 0:
        return image
    return np.roll(image, (shift.dy, shift.dx), axis=(0, 1))


# ---------------------------------------------------------------------------- extraction


@dataclass(frozen=True)
class ResidualEstimate:
    """Grain amplitude and temporal correlation, with the evidence for both."""

    sigma: float
    """Per-frame grain standard deviation, corrected for temporal correlation."""

    rho: float
    """Estimated lag-1 temporal correlation, from the lag-2 to lag-1 variance ratio."""

    legacy_sigma: float
    """What the ``sqrt(var/2)`` estimator would have reported, for comparison."""

    lag_variances: dict[int, float]
    rho_from_lag4: float
    """Independent correlation estimate from lag 4. Disagreement means the model does not fit."""

    subpixel_residual: float
    """Largest fractional offset remaining after whole-pixel alignment."""

    max_integer_shift: int
    at_bound: bool
    """A shift hit the search boundary — the window is moving more than a static window may."""

    structure: float
    alignment_applied: bool
    motion_energy: float
    """RMS of the low-pass part of the residual — structured motion that survived alignment."""

    grain_hp_std: float
    """High-pass residual amplitude, as a second line of defence against leaked motion."""

    sample_count: int

    @property
    def correlation_consistent(self) -> bool:
        """Whether lag 4 agrees with lag 1 and 2 on the correlation."""
        return abs(self.rho - self.rho_from_lag4) <= 0.10

    @property
    def correlation_trustworthy(self) -> bool:
        """Whether the reported correlation may be believed.

        Sub-pixel drift decorrelates consecutive frames, and integer alignment cannot remove it.
        A window that is both drifting *and* genuinely correlated therefore reports a correlation
        near zero — measured on synthetic material, a true rho of 0.50 combined with 0.7 px of
        drift reads back as 0.045. The amplitude survives, because the two errors partly cancel,
        but "temporal independence established" would be exactly the wrong conclusion.

        So the correlation is only meaningful on a window that is not drifting. This is the
        difference between rejecting a window and mismeasuring it.
        """
        return not self.drifting and self.correlation_consistent

    @property
    def drifting(self) -> bool:
        """Whether this window should be rejected rather than measured."""
        return self.subpixel_residual > SUBPIXEL_REJECT or self.at_bound

    @property
    def correlation_correction(self) -> float:
        """Factor by which correlation correction changed the amplitude."""
        return self.sigma / self.legacy_sigma if self.legacy_sigma > EPS else 1.0

    def as_record(self) -> dict[str, Any]:
        return {
            "sigma": self.sigma,
            "rho": self.rho,
            "legacy_sigma": self.legacy_sigma,
            "correlation_correction": self.correlation_correction,
            "rho_from_lag4": self.rho_from_lag4,
            "correlation_consistent": self.correlation_consistent,
            "correlation_trustworthy": self.correlation_trustworthy,
            "lag_variances": {str(lag): value for lag, value in self.lag_variances.items()},
            "subpixel_residual": self.subpixel_residual,
            "max_integer_shift": self.max_integer_shift,
            "at_bound": self.at_bound,
            "structure": self.structure,
            "alignment_applied": self.alignment_applied,
            "drifting": self.drifting,
            "motion_energy": self.motion_energy,
            "grain_hp_std": self.grain_hp_std,
            "sample_count": self.sample_count,
        }


def _aligned_differences(
    frames: np.ndarray,
    lag: int,
    *,
    blur_radius: int,
    motion_blur_radius: int,
    align: bool,
    max_shift: int = DEFAULT_MAX_SHIFT,
) -> tuple[np.ndarray, list[Shift], bool]:
    """Differences at the given lag, aligned only where the window is genuinely drifting.

    Alignment is decided for the window as a whole, from the **consistency** of the per-pair
    estimates, not pair by pair. Real drift produces the same shift for every pair; a featureless
    window produces estimates that scatter about zero, so their median is zero.

    That distinction matters because a spurious shift is not a harmless no-op: it destroys the
    frame-to-frame grain correlation being measured. Worse, when grain is *negatively* correlated,
    misaligning genuinely reduces residual energy — so any gate that merely rewards "less residual"
    accepts the spurious shift and then reports the correlation as near zero. Two earlier versions
    of this gate did exactly that, and the synthetic harness caught both.
    """
    estimates: list[Shift] = []
    if align:
        estimates = [
            estimate_shift(
                frames[index - lag],
                frames[index],
                blur_radius=blur_radius,
                motion_blur_radius=motion_blur_radius,
                # The same per-frame drift displaces a lag-k pair k times as far, so the search
                # bound has to scale with the lag or long-lag pairs go unaligned — which corrupts
                # the very variance ratios the correlation estimate depends on.
                max_shift=max_shift * lag,
            )
            for index in range(lag, frames.shape[0])
        ]

    # Decide for the window, not pair by pair. Medians are robust to a single pair sitting on a
    # threshold, and — more importantly — mixing aligned with unaligned differences is worse than
    # either choice: one unaligned pair contributes a motion residual that dominates the variance.
    if estimates:
        median_dy = int(np.median([shift.dy for shift in estimates]))
        median_dx = int(np.median([shift.dx for shift in estimates]))
        median_snr = float(np.median([shift.structure_snr for shift in estimates]))
        median_gain = float(np.median([shift.gain for shift in estimates]))
    else:
        median_dy = median_dx = 0
        median_snr = median_gain = 0.0

    drifting = median_dy != 0 or median_dx != 0
    align_window = (
        drifting and median_snr >= MIN_STRUCTURE_SNR and median_gain >= MIN_ALIGNMENT_GAIN
    )

    deltas: list[np.ndarray] = []
    for offset, index in enumerate(range(lag, frames.shape[0])):
        moving = frames[index]
        if align_window:
            moving = apply_shift(moving, estimates[offset])
        deltas.append(moving - frames[index - lag])
    return (np.stack(deltas) if deltas else np.empty((0,))), estimates, align_window


def _trim(deltas: np.ndarray, margin: int) -> np.ndarray:
    """Drop the border, where whole-pixel rolling wrapped content around."""
    if margin <= 0 or deltas.ndim != 3:
        return deltas
    if deltas.shape[1] <= 2 * margin or deltas.shape[2] <= 2 * margin:
        return deltas
    return deltas[:, margin:-margin, margin:-margin]


def extract(
    frames: np.ndarray,
    *,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    blur_radius: int = DEFAULT_BLUR_RADIUS,
    motion_blur_radius: int = DEFAULT_MOTION_BLUR_RADIUS,
    max_shift: int = DEFAULT_MAX_SHIFT,
    align: bool = True,
) -> ResidualEstimate:
    """Estimate grain amplitude and temporal correlation from a frame sequence.

    ``frames`` is ``(n, height, width)`` of scalar values — luma, or one channel.
    """
    stack = np.asarray(frames, dtype=np.float64)
    if stack.ndim != 3:
        raise DataError(f"expected (frames, height, width), got {stack.shape}")
    if stack.shape[0] < max(lags) + 1:
        raise DataError(
            f"need at least {max(lags) + 1} frames to measure lags {lags}; got {stack.shape[0]}"
        )

    variances: dict[int, float] = {}
    all_shifts: list[Shift] = []
    lag1_drifting = False
    lag1_deltas: np.ndarray | None = None

    for lag in sorted(lags):
        deltas, shifts, drifting = _aligned_differences(
            stack,
            lag,
            blur_radius=blur_radius,
            motion_blur_radius=motion_blur_radius,
            align=align,
            max_shift=max_shift,
        )
        applied = shifts if drifting else []
        margin = max((max(abs(shift.dy), abs(shift.dx)) for shift in applied), default=0)
        trimmed = _trim(deltas, margin)
        variances[lag] = float(np.var(trimmed))
        if lag == 1:
            lag1_deltas = trimmed
            all_shifts = shifts
            lag1_drifting = drifting

    if lag1_deltas is None or lag1_deltas.size == 0:
        raise DataError("no lag-1 differences could be formed")

    var1 = variances.get(1, 0.0)
    var2 = variances.get(2)
    var4 = variances.get(4)

    # rho from the lag-2 to lag-1 variance ratio: Var(d2)/Var(d1) = 1 + rho.
    rho = float(np.clip(var2 / var1 - 1.0, -0.99, 0.99)) if var2 is not None and var1 > EPS else 0.0

    # Lag 4 must satisfy Var(d4)/Var(d1) = 1 + rho + rho^2 + rho^3. Solve numerically for an
    # independent estimate; disagreement means the AR(1) model does not describe this window.
    rho_from_lag4 = rho
    if var4 is not None and var1 > EPS:
        target = var4 / var1
        candidates = np.linspace(-0.99, 0.99, 1991)
        predicted = 1.0 + candidates + candidates**2 + candidates**3
        rho_from_lag4 = float(candidates[int(np.argmin(np.abs(predicted - target)))])

    legacy_sigma = float(np.sqrt(max(var1, 0.0) / 2.0))
    sigma = float(np.sqrt(max(var1, 0.0) / (2.0 * max(1.0 - rho, EPS))))

    low_pass = np.stack([box_blur(delta, motion_blur_radius) for delta in lag1_deltas])
    high_pass = lag1_deltas - low_pass

    return ResidualEstimate(
        sigma=sigma,
        rho=rho,
        legacy_sigma=legacy_sigma,
        lag_variances=variances,
        rho_from_lag4=rho_from_lag4,
        subpixel_residual=max((shift.subpixel_magnitude for shift in all_shifts), default=0.0),
        at_bound=any(shift.at_bound for shift in all_shifts),
        max_integer_shift=max(
            (max(abs(shift.dy), abs(shift.dx)) for shift in all_shifts if lag1_drifting),
            default=0,
        ),
        structure=float(np.mean([shift.structure for shift in all_shifts])) if all_shifts else 0.0,
        alignment_applied=bool(lag1_drifting),
        motion_energy=float(np.sqrt(np.mean(low_pass**2))),
        grain_hp_std=float(np.sqrt(np.mean(high_pass**2) / (2.0 * max(1.0 - rho, EPS)))),
        sample_count=int(lag1_deltas.size),
    )


__all__ = [
    "DEFAULT_BLUR_RADIUS",
    "DEFAULT_LAGS",
    "DEFAULT_MAX_SHIFT",
    "DEFAULT_MOTION_BLUR_RADIUS",
    "MIN_ALIGNMENT_GAIN",
    "MIN_STRUCTURE_SNR",
    "SUBPIXEL_REJECT",
    "ResidualEstimate",
    "Shift",
    "apply_shift",
    "box_blur",
    "estimate_shift",
    "extract",
]
