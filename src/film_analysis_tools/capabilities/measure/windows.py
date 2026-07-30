"""Window selection: finding measurable regions, and being honest about what was not found.

Grain amplitude can only be measured where the picture is static. The legacy tool knew this and
said so in its help text — that raising the motion gate "mines the least-moving background at
higher contamination risk", and that if nothing passes, the scene is not a grain source. Good
advice, but advice: nothing stopped a run from widening the gate for yield, and nothing made a
thin result look thin.

The consequence is visible in the shipped legacy preset. Its amplitude curve rests on 24 points
spanning linear luma **0.00012 to 0.282** — six knots, the brightest at 0.222. Four fifths of the
curve's domain has no measurement behind it at all, because the windows that survived the motion
gate happened to be the dark ones. The preset records its coverage honestly, but only after the
fact, and nothing in the selection stage ever said "you have no highlights".

So two things are structural here rather than advisory:

* **The gate never relaxes itself.** A caller may supply a different gate, but the gate used is
  recorded in the report and compared against the default, so a widened gate is visible in any
  artifact derived from it.
* **Coverage is a result, not a diagnostic.** The report knows which strata are empty and says so,
  and :attr:`SelectionReport.sufficient` is false when the material cannot support a curve —
  before anything is fitted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from film_analysis_tools.capabilities.measure.admissibility import OverlayEvidence
from film_analysis_tools.capabilities.measure.residual import (
    DEFAULT_BLUR_RADIUS,
    DEFAULT_MOTION_BLUR_RADIUS,
    MIN_STRUCTURE_SNR,
    box_blur,
    estimate_shift,
)
from film_analysis_tools.core.errors import DataError, SelectionError

#: Luma band edges on the working linear scale. Absolute rather than per-scene quantiles: the
#: point is to cover the *amplitude curve's domain*, and quantile bands would report full coverage
#: on a scene that is uniformly dark.
DEFAULT_BAND_EDGES: tuple[float, float] = (0.02, 0.25)
BANDS: tuple[str, ...] = ("shadow", "midtone", "highlight")
TEXTURES: tuple[str, ...] = ("flat", "textured")
POSITIONS: tuple[str, ...] = ("centre", "edge")


@dataclass(frozen=True)
class WindowGate:
    """Acceptance thresholds. Frozen, and recorded with every result that depends on them."""

    max_motion_energy: float = 0.005
    """RMS of the low-pass temporal difference. A genuinely static region sits near 0.001-0.003."""

    max_subpixel_residual: float = 0.25
    """Fractional drift left after whole-pixel alignment. Beyond this the window is moving."""

    min_frames: int = 5
    min_windows_per_band: int = 3
    min_bands: int = 2

    def as_record(self) -> dict[str, Any]:
        return {
            "max_motion_energy": self.max_motion_energy,
            "max_subpixel_residual": self.max_subpixel_residual,
            "min_frames": self.min_frames,
            "min_windows_per_band": self.min_windows_per_band,
            "min_bands": self.min_bands,
        }


DEFAULT_GATE = WindowGate()


@dataclass(frozen=True)
class Window:
    """One candidate region, with the measurements the gate is applied to."""

    x: int
    y: int
    size: int
    level: float
    motion_energy: float
    structure_snr: float
    subpixel_residual: float
    band: str
    texture: str
    position: str
    level_low: float = 0.0
    level_high: float = 0.0
    """1st and 99th percentile inside the tile, over its frames.

    A tile mean is the same "judge by the average" statistic that made the whole film read as
    midtone at interval level. Measured on the 4K master: of 120 tiles in one interval, 18 reach
    the highlight edge by mean and **32** by content — a lamp inside a 128 px tile averages away.
    """

    @property
    def stratum(self) -> tuple[str, str, str]:
        return (self.band, self.texture, self.position)

    def contains_band(self, band: str, edges: tuple[float, float] = DEFAULT_BAND_EDGES) -> bool:
        """Whether the tile *contains* content in a band, by its spread rather than its mean."""
        low, high = edges
        if band == "shadow":
            return self.level_low < low
        if band == "midtone":
            return self.level_low < high and self.level_high >= low
        if band == "highlight":
            return self.level_high >= high
        raise SelectionError(f"unknown band {band!r}; expected one of {BANDS}")

    def slice_of(self, frames: np.ndarray) -> np.ndarray:
        return frames[:, self.y : self.y + self.size, self.x : self.x + self.size]

    def as_record(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "size": self.size,
            "level": self.level,
            "motion_energy": self.motion_energy,
            "structure_snr": self.structure_snr,
            "subpixel_residual": self.subpixel_residual,
            "band": self.band,
            "texture": self.texture,
            "position": self.position,
            "level_low": self.level_low,
            "level_high": self.level_high,
        }


@dataclass(frozen=True)
class Rejection:
    window: Window
    reason: str


@dataclass(frozen=True)
class SelectionReport:
    """What was accepted, what was rejected and why, and what the material cannot support."""

    accepted: tuple[Window, ...]
    rejected: tuple[Rejection, ...]
    gate: WindowGate
    band_edges: tuple[float, float]
    frames: int
    notes: tuple[str, ...] = field(default=())

    @property
    def gate_is_default(self) -> bool:
        """False means a caller widened the gate. Any artifact derived from this must say so."""
        return self.gate == DEFAULT_GATE

    def coverage(self) -> dict[str, int]:
        counts = dict.fromkeys(BANDS, 0)
        for window in self.accepted:
            counts[window.band] += 1
        return counts

    def texture_coverage(self) -> dict[str, int]:
        counts = dict.fromkeys(TEXTURES, 0)
        for window in self.accepted:
            counts[window.texture] += 1
        return counts

    def missing_bands(self) -> list[str]:
        """Bands with too few windows to support a curve knot."""
        counts = self.coverage()
        return [band for band in BANDS if counts[band] < self.gate.min_windows_per_band]

    def measured_level_range(self) -> tuple[float, float]:
        if not self.accepted:
            return (0.0, 0.0)
        levels = [window.level for window in self.accepted]
        return (min(levels), max(levels))

    @property
    def sufficient(self) -> bool:
        """Whether the material can support an amplitude curve at all.

        Checked *before* fitting, so a run that found only shadows stops here rather than
        producing a curve that looks complete and is flat everywhere it was never measured.
        """
        populated = len(BANDS) - len(self.missing_bands())
        return bool(self.accepted) and populated >= self.gate.min_bands

    def rejection_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rejection in self.rejected:
            counts[rejection.reason] = counts.get(rejection.reason, 0) + 1
        return counts

    def as_record(self) -> dict[str, Any]:
        low, high = self.measured_level_range()
        return {
            "frames": self.frames,
            "gate": self.gate.as_record(),
            "gate_is_default": self.gate_is_default,
            "band_edges": list(self.band_edges),
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "coverage": self.coverage(),
            "texture_coverage": self.texture_coverage(),
            "missing_bands": self.missing_bands(),
            "measured_level_min": low,
            "measured_level_max": high,
            "sufficient": self.sufficient,
            "rejection_reasons": self.rejection_reasons(),
            "windows": [window.as_record() for window in self.accepted],
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        low, high = self.measured_level_range()
        lines = [
            f"windows: {len(self.accepted)} accepted, {len(self.rejected)} rejected"
            f"{'' if self.gate_is_default else '   [GATE WIDENED]'}",
            f"levels : {low:.5f} to {high:.5f}",
            "bands  : " + ", ".join(f"{name} {count}" for name, count in self.coverage().items()),
        ]
        for reason, count in sorted(self.rejection_reasons().items(), key=lambda x: -x[1]):
            lines.append(f"  rejected {count:4d}  {reason}")
        missing = self.missing_bands()
        if not self.sufficient:
            lines.append(
                "INSUFFICIENT: this material cannot support an amplitude curve "
                f"(usable bands: {len(BANDS) - len(missing)}, need {self.gate.min_bands}). "
                "Widening the gate mines moving background; shoot or select different material."
            )
        elif missing:
            lines.append(
                f"PARTIAL COVERAGE: too few windows in {', '.join(missing)} — a curve fitted "
                "here is unmeasured there and must not be presented as if it were measured."
            )
        return "\n".join(lines)


def _band_of(level: float, edges: tuple[float, float]) -> str:
    if level < edges[0]:
        return "shadow"
    return "midtone" if level < edges[1] else "highlight"


def _position_of(x: int, y: int, size: int, width: int, height: int) -> str:
    centre_x, centre_y = x + size / 2, y + size / 2
    inside_x = width * 0.25 <= centre_x <= width * 0.75
    inside_y = height * 0.25 <= centre_y <= height * 0.75
    return "centre" if inside_x and inside_y else "edge"


def select_windows(
    frames: np.ndarray,
    *,
    gate: WindowGate = DEFAULT_GATE,
    size: int = 128,
    stride: int | None = None,
    band_edges: tuple[float, float] = DEFAULT_BAND_EDGES,
    blur_radius: int = DEFAULT_BLUR_RADIUS,
    motion_blur_radius: int = DEFAULT_MOTION_BLUR_RADIUS,
    overlay: OverlayEvidence | None = None,
) -> SelectionReport:
    """Find measurable windows across level, texture and frame position.

    Never widens the gate to increase yield. A caller who supplies a wider gate gets it recorded
    in the report, and every artifact derived from that report can see it.
    """
    stack = np.asarray(frames, dtype=np.float64)
    if stack.ndim != 3:
        raise DataError(f"expected (frames, height, width), got {stack.shape}")
    if stack.shape[0] < gate.min_frames:
        raise DataError(
            f"need at least {gate.min_frames} frames for a stable window measurement; "
            f"got {stack.shape[0]}"
        )

    count, height, width = stack.shape
    if size > min(height, width):
        raise DataError(f"window size {size} exceeds the frame ({height}x{width})")
    step = stride or size

    # Low-pass temporal difference: structured motion survives blurring, grain does not.
    deltas = stack[1:] - stack[:-1]
    low_pass = np.stack([box_blur(delta, motion_blur_radius) for delta in deltas])

    accepted: list[Window] = []
    rejected: list[Rejection] = []

    for y in range(0, height - size + 1, step):
        for x in range(0, width - size + 1, step):
            region = stack[:, y : y + size, x : x + size]
            motion = float(np.sqrt(np.mean(low_pass[:, y : y + size, x : x + size] ** 2)))
            shift = estimate_shift(region[0], region[1], blur_radius=blur_radius)
            level = float(np.mean(region))
            spread_low, spread_high = (float(v) for v in np.percentile(region, (1.0, 99.0)))

            window = Window(
                x=x,
                y=y,
                size=size,
                level=level,
                motion_energy=motion,
                structure_snr=shift.structure_snr,
                subpixel_residual=shift.subpixel_magnitude,
                band=_band_of(level, band_edges),
                texture="textured" if shift.structure_snr >= MIN_STRUCTURE_SNR else "flat",
                position=_position_of(x, y, size, width, height),
                level_low=spread_low,
                level_high=spread_high,
            )

            # Sub-pixel drift only means something where there is structure to misalign. On a
            # flat window the shift estimate is noise-driven, including its fractional part, and
            # rejecting on it would discard exactly the windows best suited to measuring grain.
            drift_is_meaningful = shift.structure_snr >= MIN_STRUCTURE_SNR

            if overlay is not None and overlay.excludes(x, y, size):
                rejected.append(Rejection(window, "overlaps a composited region"))
            elif motion > gate.max_motion_energy:
                rejected.append(Rejection(window, "motion above gate"))
            elif drift_is_meaningful and window.subpixel_residual > gate.max_subpixel_residual:
                rejected.append(Rejection(window, "sub-pixel drift"))
            else:
                accepted.append(window)

    notes: list[str] = []
    if not accepted:
        notes.append(
            "no window passed the gate: this material is not a grain-amplitude source. "
            "Widening the gate mines moving background and contaminates the measurement."
        )
    return SelectionReport(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        gate=gate,
        band_edges=band_edges,
        frames=count,
        notes=tuple(notes),
    )


def stratified(report: SelectionReport, *, per_stratum: int = 2) -> list[Window]:
    """A spread across strata rather than whichever windows happened to score best.

    Taking the top-scoring windows is how a corpus ends up measured only in the shadows: the
    quietest regions of most footage are the dark ones.
    """
    grouped: dict[tuple[str, str, str], list[Window]] = {}
    for window in report.accepted:
        grouped.setdefault(window.stratum, []).append(window)
    chosen: list[Window] = []
    for stratum in sorted(grouped):
        ordered = sorted(grouped[stratum], key=lambda window: window.motion_energy)
        chosen.extend(ordered[:per_stratum])
    return chosen


def with_gate(report: SelectionReport, gate: WindowGate) -> SelectionReport:
    """Re-label a report with a different gate. Does not re-select — used only in tests."""
    return replace(report, gate=gate)


def as_records(windows: Sequence[Window]) -> list[dict[str, Any]]:
    return [window.as_record() for window in windows]


__all__ = [
    "BANDS",
    "DEFAULT_BAND_EDGES",
    "DEFAULT_GATE",
    "POSITIONS",
    "TEXTURES",
    "Rejection",
    "SelectionReport",
    "Window",
    "WindowGate",
    "as_records",
    "select_windows",
    "stratified",
    "with_gate",
]
