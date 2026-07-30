"""Scene admissibility: the three checks the catalogue could not make.

`docs/scene-selection-criteria.md` lists eight criteria a scene must satisfy before its grain can
be measured. Five were already computable from the legacy catalogue. Three were not, and each
corrupts a different measurement in a way no existing gate detects:

* **A3 — fades, dissolves and exposure ramps.** The estimator assumes stationarity. A level ramp
  puts a deterministic offset into *every* temporal difference. A uniform level change is entirely
  low-frequency, so the motion gate *does* see it — but only as an absolute quantity, and cannot
  know how damaging it is. Measured against a fixed 0.005 gate, the same 0.003-per-frame ramp
  contributes 1% of difference variance at sigma 0.02 and **100% at sigma 0.002**, and the gate
  passes it in both cases. Expressed as a share of the variance being measured, A3 scales with the
  grain: it catches exactly the ramps that matter on clean, low-amplitude material — which is the
  highlight end the corpus is already short of.
* **A6 — clipping.** ``luma_peak_code`` records how bright a scene got; it does not record how
  much of the picture sat *at* that limit. Clipping truncates the residual distribution: sigma
  biases low, kurtosis biases negative, and the tail evidence becomes meaningless while still
  looking like a number.
* **A7 — overlays and composited graphics.** Titles and subtitles carry no negative grain. A
  region with no noise pulls the measured amplitude down, and unlike motion it is perfectly
  static, so every staticness gate welcomes it.

Each is measured as a **share of the quantity it would corrupt**, not in absolute units, so the
thresholds do not need recalibrating per source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from film_analysis_tools.core.errors import DataError

EPS = 1.0e-12

#: Share of temporal-difference variance a whole-frame level change may contribute before the
#: scene counts as ramping. Chosen against measurement: a static scene sits near zero, and a fade
#: slow enough to be invisible on screen already dominates the residual.
MAX_RAMP_SHARE = 0.05

#: Share of pixels allowed at the ceiling or floor. Clipping does not degrade gracefully — a few
#: percent is enough to truncate the tail the distribution evidence depends on.
MAX_CLIPPED_FRACTION = 0.02

#: Share of the frame allowed to carry essentially no temporal noise. Real grain is everywhere;
#: a noise-free region is composited, and including it drags the measured amplitude down.
MAX_NOISE_FREE_FRACTION = 0.05

#: A block counts as noise-free below this share of the frame's median block residual.
NOISE_FREE_RATIO = 0.25


# ------------------------------------------------------------------ A3: fades and ramps


@dataclass(frozen=True)
class RampEvidence:
    """Whole-frame level change across the sequence."""

    slope_per_frame: float
    """Mean level change per frame, in working units."""

    variance_share: float
    """Share of temporal-difference variance explained by whole-frame level change.

    This is the number that matters: the ramp contributes ``mean(delta)**2`` to a variance whose
    grain part is ``2 sigma**2``, so the share says directly how much of the measured amplitude
    would be the fade rather than the film.
    """

    linearity: float
    """R-squared of a straight-line fit to per-frame mean level. High means a steady fade;
    low with a large slope means flicker or a lighting change."""

    @property
    def is_ramping(self) -> bool:
        return self.variance_share > MAX_RAMP_SHARE

    def as_record(self) -> dict[str, Any]:
        return {
            "slope_per_frame": self.slope_per_frame,
            "variance_share": self.variance_share,
            "linearity": self.linearity,
            "is_ramping": self.is_ramping,
        }


def ramp_evidence(frames: np.ndarray) -> RampEvidence:
    """Detect a fade, dissolve or exposure ramp.

    Works on the *mean* of each temporal difference. Grain has zero mean by construction, so any
    systematic offset in the per-difference mean is a whole-frame level change.

    This does not replace the motion gate — a uniform ramp is low-frequency and the gate sees it
    too. It replaces the gate's *absolute* judgement with a relative one, and names the cause: the
    gate can only say "too much motion", where this says "a fade is contributing this share of the
    variance you are about to call grain".
    """
    stack = np.asarray(frames, dtype=np.float64)
    if stack.ndim != 3 or stack.shape[0] < 3:
        raise DataError(f"ramp evidence needs at least 3 frames, got {stack.shape}")

    per_frame_mean = stack.reshape(stack.shape[0], -1).mean(axis=1)
    deltas = stack[1:] - stack[:-1]
    difference_means = deltas.reshape(deltas.shape[0], -1).mean(axis=1)

    slope = float(np.mean(difference_means))
    total_variance = float(np.var(deltas))
    variance_share = (slope**2) / total_variance if total_variance > EPS else 0.0

    index = np.arange(per_frame_mean.size, dtype=np.float64)
    spread = float(np.var(per_frame_mean))
    if spread > EPS:
        fit = np.polyfit(index, per_frame_mean, 1)
        residual = per_frame_mean - np.polyval(fit, index)
        linearity = float(1.0 - np.var(residual) / spread)
    else:
        linearity = 0.0

    return RampEvidence(
        slope_per_frame=slope,
        variance_share=float(min(variance_share, 1.0)),
        linearity=linearity,
    )


# ---------------------------------------------------------------------- A6: clipping


@dataclass(frozen=True)
class ClippingEvidence:
    """How much of the picture sits at the format limits."""

    high_fraction: float
    low_fraction: float
    ceiling: float
    floor: float

    @property
    def total_fraction(self) -> float:
        return self.high_fraction + self.low_fraction

    @property
    def is_clipped(self) -> bool:
        return self.total_fraction > MAX_CLIPPED_FRACTION

    def as_record(self) -> dict[str, Any]:
        return {
            "high_fraction": self.high_fraction,
            "low_fraction": self.low_fraction,
            "total_fraction": self.total_fraction,
            "ceiling": self.ceiling,
            "floor": self.floor,
            "is_clipped": self.is_clipped,
        }


def clipping_evidence(
    frames: np.ndarray,
    *,
    ceiling: float = 1.0,
    floor: float = 0.0,
    tolerance: float = 1.0 / 1023.0,
) -> ClippingEvidence:
    """Fraction of samples at the ceiling or floor.

    The peak code alone cannot answer this. A scene that touches the ceiling in one specular
    highlight and one that is blown across half the frame report the same peak, and only the
    second destroys the distribution evidence.
    """
    stack = np.asarray(frames, dtype=np.float64)
    if stack.size == 0:
        raise DataError("clipping evidence needs samples")
    return ClippingEvidence(
        high_fraction=float(np.mean(stack >= ceiling - tolerance)),
        low_fraction=float(np.mean(stack <= floor + tolerance)),
        ceiling=ceiling,
        floor=floor,
    )


# ---------------------------------------------------------------------- A7: overlays


@dataclass(frozen=True)
class OverlayEvidence:
    """Regions carrying no temporal noise — composited rather than photographed."""

    noise_free_fraction: float
    median_block_residual: float
    block_size: int
    noise_free_blocks: np.ndarray
    """Block grid, true where the region carries essentially no temporal noise.

    A small logo should not disqualify a whole scene — it should stop a *window* being placed
    on it. The mask is what lets selection do that, so the scene-level flag is reserved for
    material that is largely graphics.
    """

    @property
    def has_overlay(self) -> bool:
        """Whether overlays are extensive enough to disqualify the scene outright."""
        return self.noise_free_fraction > MAX_NOISE_FREE_FRACTION

    @property
    def any_overlay(self) -> bool:
        """Whether any region should be excluded from window placement."""
        return bool(self.noise_free_blocks.any())

    def excludes(self, x: int, y: int, size: int) -> bool:
        """Whether a window at ``(x, y)`` would overlap a noise-free region."""
        rows, columns = self.noise_free_blocks.shape
        y0, y1 = y // self.block_size, min(rows, (y + size - 1) // self.block_size + 1)
        x0, x1 = x // self.block_size, min(columns, (x + size - 1) // self.block_size + 1)
        if y0 >= rows or x0 >= columns:
            return False
        return bool(self.noise_free_blocks[y0:y1, x0:x1].any())

    def as_record(self) -> dict[str, Any]:
        return {
            "noise_free_fraction": self.noise_free_fraction,
            "median_block_residual": self.median_block_residual,
            "block_size": self.block_size,
            "has_overlay": self.has_overlay,
            "any_overlay": self.any_overlay,
            "noise_free_block_count": int(self.noise_free_blocks.sum()),
        }


def overlay_evidence(frames: np.ndarray, *, block_size: int = 32) -> OverlayEvidence:
    """Detect composited graphics by the noise they do not have.

    Film grain is present everywhere in a photographed frame. A title, subtitle or logo added
    after the scan is not, so its temporal residual is essentially zero. That is the reliable
    signature — far more so than looking for sharp edges or particular shapes — and it is the
    property that matters, because a noise-free region included in a window drags the measured
    amplitude down while passing every staticness test.
    """
    stack = np.asarray(frames, dtype=np.float64)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise DataError(f"overlay evidence needs at least 2 frames, got {stack.shape}")

    deltas = stack[1:] - stack[:-1]
    _, height, width = deltas.shape
    rows, columns = height // block_size, width // block_size
    if rows < 2 or columns < 2:
        raise DataError(f"frame {height}x{width} is too small for {block_size}px blocks")

    trimmed = deltas[:, : rows * block_size, : columns * block_size]
    blocks = trimmed.reshape(trimmed.shape[0], rows, block_size, columns, block_size)
    residual = np.sqrt((blocks**2).mean(axis=(0, 2, 4)))

    median = float(np.median(residual))
    if median <= EPS:
        return OverlayEvidence(
            noise_free_fraction=1.0,
            median_block_residual=median,
            block_size=block_size,
            noise_free_blocks=np.ones_like(residual, dtype=bool),
        )
    mask = residual < median * NOISE_FREE_RATIO
    return OverlayEvidence(
        noise_free_fraction=float(np.mean(mask)),
        median_block_residual=median,
        block_size=block_size,
        noise_free_blocks=mask,
    )


# ------------------------------------------------------------------------- combined


@dataclass(frozen=True)
class SceneAdmissibility:
    """Whether a scene may be measured, and what disqualifies it."""

    ramp: RampEvidence
    clipping: ClippingEvidence
    overlay: OverlayEvidence

    @property
    def reasons(self) -> tuple[str, ...]:
        found: list[str] = []
        if self.ramp.is_ramping:
            found.append(
                f"level ramp contributes {self.ramp.variance_share:.1%} of difference variance"
            )
        if self.clipping.is_clipped:
            found.append(f"{self.clipping.total_fraction:.1%} of samples clipped")
        if self.overlay.has_overlay:
            found.append(
                f"{self.overlay.noise_free_fraction:.1%} of the frame carries no temporal noise"
            )
        return tuple(found)

    @property
    def admissible(self) -> bool:
        return not self.reasons

    def as_record(self) -> dict[str, Any]:
        return {
            "admissible": self.admissible,
            "reasons": list(self.reasons),
            "ramp": self.ramp.as_record(),
            "clipping": self.clipping.as_record(),
            "overlay": self.overlay.as_record(),
        }

    def summary(self) -> str:
        if self.admissible:
            return "admissible"
        return "not admissible: " + "; ".join(self.reasons)


def scene_admissibility(
    frames: np.ndarray,
    *,
    ceiling: float = 1.0,
    floor: float = 0.0,
    block_size: int = 32,
) -> SceneAdmissibility:
    """All three checks the catalogue could not make."""
    return SceneAdmissibility(
        ramp=ramp_evidence(frames),
        clipping=clipping_evidence(frames, ceiling=ceiling, floor=floor),
        overlay=overlay_evidence(frames, block_size=block_size),
    )


__all__ = [
    "MAX_CLIPPED_FRACTION",
    "MAX_NOISE_FREE_FRACTION",
    "MAX_RAMP_SHARE",
    "ClippingEvidence",
    "OverlayEvidence",
    "RampEvidence",
    "SceneAdmissibility",
    "clipping_evidence",
    "overlay_evidence",
    "ramp_evidence",
    "scene_admissibility",
]
