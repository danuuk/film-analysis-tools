"""Per-measurement scene screening: which scenes may be used for which measurement, and why not.

Encodes sections 3 and 4 of ``docs/scene-selection-criteria.md``. Screening runs on scene
*metadata* — no decoding — so a whole catalogue can be filtered before a single frame is read.
Frame-level checks that need pixels (A3 ramp, A6 clipped fraction, A7 overlays) live in
``admissibility`` and run on the survivors.

Two things this is deliberately not.

It is **not a ranking.** Scenes are admissible or they are not; there is no score to sort by and
take the top of. Ranking is how a corpus ends up measured only where the ranking pointed, and the
catalogue's own ``grain_score`` — which ranks by how grainy a scene *looks* — would bias the
amplitude curve upward exactly where it has fewest points.

It is **not one filter.** A scene admissible for amplitude may be useless for the spectrum, and
the requirements genuinely conflict: the spectrum needs flat windows while drift detection needs
textured ones. Screening therefore takes the measurement as an argument and answers for that
measurement alone.

The scene descriptor here is FAT-native. Populating it from any particular catalogue is an import
step, so nothing in the screening logic depends on how some other project happened to organise its
metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from film_analysis_tools.core.errors import SelectionError


class Measurement(Enum):
    """The five evidence types, each with its own admissibility requirements."""

    AMPLITUDE = "amplitude_vs_level"
    SPECTRUM = "spectrum"
    DISTRIBUTION = "distribution"
    HETEROGENEITY = "slow_heterogeneity"
    TEMPORAL = "temporal"


@dataclass(frozen=True)
class SceneCandidate:
    """What screening needs to know about a scene, before any frame is decoded."""

    scene_id: str
    duration_s: float
    frame_count: int
    static_score: float
    """1.0 is perfectly still. Criterion A1."""

    cut_score: float
    """Likelihood of a cut inside the span. Criterion A2."""

    signal_bit_depth: float | None
    """Effective bit depth of the signal. Criterion A5. ``None`` when the catalogue does not
    measure it — some record the container's depth, which is a constant and says nothing."""

    peak_code: float | None
    floor_code: float | None
    """Extremes actually reached, normalised to 0..1. Criterion A6, partially — the *fraction* at
    the limit needs pixels. ``None`` when the catalogue records the legal-range limits instead of
    measured extremes, which is common and is not the same thing."""

    median_level: float
    level_p10: float
    level_p90: float
    """Level distribution, used for coverage potential rather than admissibility."""

    resampled: bool = False
    """Whether the decode path scales. Criterion A8 — a source-record property, not a scene one."""

    labels: tuple[str, ...] = ()

    @property
    def level_span(self) -> float:
        return max(0.0, self.level_p90 - self.level_p10)


@dataclass(frozen=True)
class ScreeningThresholds:
    """Where each criterion cuts. Recorded with every report that depends on them."""

    min_static_score: float = 0.7
    max_cut_score: float = 0.3
    min_frames: int = 8
    """Lag 4 needs 5; a *stable* variance ratio needs more, and too few frames fails quietly by
    producing a noisy correlation that looks like a measurement."""

    min_bit_depth: float = 9.0
    """Below this the residual measures the quantiser rather than the grain."""

    max_peak_code: float = 0.999
    min_floor_code: float = 0.001

    def as_record(self) -> dict[str, Any]:
        return {
            "min_static_score": self.min_static_score,
            "max_cut_score": self.max_cut_score,
            "min_frames": self.min_frames,
            "min_bit_depth": self.min_bit_depth,
            "max_peak_code": self.max_peak_code,
            "min_floor_code": self.min_floor_code,
        }


DEFAULT_THRESHOLDS = ScreeningThresholds()

#: Which criteria each measurement requires. From section 4 of the criteria document — the
#: differences are the point, so they are stated per measurement rather than merged into one gate.
REQUIREMENTS: Mapping[Measurement, frozenset[str]] = {
    # Needs static windows across the level range; quantisation and clipping both distort the
    # amplitude it is trying to read.
    Measurement.AMPLITUDE: frozenset({"A1", "A2", "A4", "A5", "A6"}),
    # Any resampling makes the measured spectrum the scaler's, so A8 is absolute here.
    # Quantisation likewise imprints its own structure on the noise power.
    Measurement.SPECTRUM: frozenset({"A1", "A2", "A4", "A5", "A8"}),
    # Clipping is the critical one: it truncates the tail this measurement exists to describe.
    Measurement.DISTRIBUTION: frozenset({"A1", "A2", "A4", "A5", "A6"}),
    # The lightest requirements — it compares slow structure across sources rather than measuring
    # amplitude — but it needs two scenes, which is a set property checked separately.
    Measurement.HETEROGENEITY: frozenset({"A1", "A2"}),
    # Needs enough frames for lag 4 above all, and stillness so drift cannot mask correlation.
    Measurement.TEMPORAL: frozenset({"A1", "A2", "A4"}),
}

#: Sentinel returned by a criterion this catalogue cannot answer.
UNASSESSABLE = "\x00unassessable"

CRITERION_NAMES: Mapping[str, str] = {
    "A1": "static enough",
    "A2": "no cut inside the span",
    "A4": "enough frames for lag 4",
    "A5": "no heavy quantisation",
    "A6": "not clipped at the format limits",
    "A8": "decode path free of resampling",
}


@dataclass(frozen=True)
class SceneVerdict:
    """Whether one scene may be used for one measurement."""

    scene: SceneCandidate
    measurement: Measurement
    failed: tuple[tuple[str, str], ...]
    """``(criterion, why)`` pairs for criteria the scene definitively fails."""

    unassessed: tuple[str, ...] = ()
    """Required criteria this catalogue cannot answer.

    Kept separate from a pass on purpose. A metric that is a constant for every scene — a
    container's bit depth, a format's legal-range limits — silently satisfies any threshold while
    measuring nothing, and a scene admitted that way looks identical to one that was checked.
    """

    @property
    def rejected(self) -> bool:
        return bool(self.failed)

    @property
    def needs_frame_check(self) -> bool:
        """Passed everything the catalogue can answer, but not everything required."""
        return not self.failed and bool(self.unassessed)

    @property
    def admissible(self) -> bool:
        """Fully screened and passed. Excludes scenes still needing a frame-level check."""
        return not self.failed and not self.unassessed

    @property
    def usable(self) -> bool:
        """Not rejected — admissible, or admissible pending the frame-level check."""
        return not self.failed

    def as_record(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene.scene_id,
            "measurement": self.measurement.value,
            "admissible": self.admissible,
            "needs_frame_check": self.needs_frame_check,
            "failed": [{"criterion": key, "why": why} for key, why in self.failed],
            "unassessed": list(self.unassessed),
        }


def _check(scene: SceneCandidate, criterion: str, thresholds: ScreeningThresholds) -> str | None:
    """Return why the criterion fails, or ``None`` when it passes."""
    if criterion == "A5" and scene.signal_bit_depth is None:
        return UNASSESSABLE
    if criterion == "A6" and (scene.peak_code is None or scene.floor_code is None):
        return UNASSESSABLE
    if criterion == "A1":
        if scene.static_score < thresholds.min_static_score:
            return f"static_score {scene.static_score:.2f} < {thresholds.min_static_score}"
    elif criterion == "A2":
        if scene.cut_score > thresholds.max_cut_score:
            return f"cut_score {scene.cut_score:.2f} > {thresholds.max_cut_score}"
    elif criterion == "A4":
        if scene.frame_count < thresholds.min_frames:
            return f"{scene.frame_count} frames < {thresholds.min_frames}"
    elif criterion == "A5":
        assert scene.signal_bit_depth is not None
        if scene.signal_bit_depth < thresholds.min_bit_depth:
            return f"signal bit depth {scene.signal_bit_depth:.1f} < {thresholds.min_bit_depth}"
    elif criterion == "A6":
        assert scene.peak_code is not None and scene.floor_code is not None
        if scene.peak_code > thresholds.max_peak_code:
            return f"peak code {scene.peak_code:.4f} at the ceiling"
        if scene.floor_code < thresholds.min_floor_code:
            return f"floor code {scene.floor_code:.4f} at the floor"
    elif criterion == "A8":
        if scene.resampled:
            return "decode path resamples, which correlates noise spatially"
    else:
        raise SelectionError(f"unknown criterion {criterion!r}")
    return None


def screen_scene(
    scene: SceneCandidate,
    measurement: Measurement,
    *,
    thresholds: ScreeningThresholds = DEFAULT_THRESHOLDS,
) -> SceneVerdict:
    """Whether one scene is admissible for one measurement."""
    failed: list[tuple[str, str]] = []
    unassessed: list[str] = []
    for criterion in sorted(REQUIREMENTS[measurement]):
        why = _check(scene, criterion, thresholds)
        if why is UNASSESSABLE:
            unassessed.append(criterion)
        elif why is not None:
            failed.append((criterion, why))
    return SceneVerdict(
        scene=scene,
        measurement=measurement,
        failed=tuple(failed),
        unassessed=tuple(unassessed),
    )


@dataclass(frozen=True)
class ScreeningReport:
    """The whole catalogue, screened for one measurement."""

    measurement: Measurement
    verdicts: tuple[SceneVerdict, ...]
    thresholds: ScreeningThresholds

    @property
    def admissible(self) -> tuple[SceneCandidate, ...]:
        """Fully screened and passed."""
        return tuple(v.scene for v in self.verdicts if v.admissible)

    @property
    def usable(self) -> tuple[SceneCandidate, ...]:
        """Not rejected — the population the frame-level checks should run on."""
        return tuple(v.scene for v in self.verdicts if v.usable)

    @property
    def needs_frame_check(self) -> tuple[SceneCandidate, ...]:
        return tuple(v.scene for v in self.verdicts if v.needs_frame_check)

    def unassessed_criteria(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for verdict in self.verdicts:
            for criterion in verdict.unassessed:
                key = f"{criterion} {CRITERION_NAMES[criterion]}"
                counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def thresholds_are_default(self) -> bool:
        return self.thresholds == DEFAULT_THRESHOLDS

    def rejection_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for verdict in self.verdicts:
            for criterion, _why in verdict.failed:
                key = f"{criterion} {CRITERION_NAMES[criterion]}"
                counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))

    def coverage_potential(self, band_edges: tuple[float, float] = (0.02, 0.25)) -> dict[str, int]:
        """How many admissible scenes could contribute a window in each band.

        Uses each scene's *level span*, not its mean. A scene whose p10 sits in shadow and whose
        p90 sits in highlight can supply windows in both, and judging it by its average would
        discard that — which is how a catalogue of 304 scenes reads as almost entirely midtone.
        """
        counts = {"shadow": 0, "midtone": 0, "highlight": 0}
        low, high = band_edges
        for scene in self.usable:
            if scene.level_p10 < low:
                counts["shadow"] += 1
            if scene.level_p10 < high and scene.level_p90 >= low:
                counts["midtone"] += 1
            if scene.level_p90 >= high:
                counts["highlight"] += 1
        return counts

    def as_record(self) -> dict[str, Any]:
        return {
            "measurement": self.measurement.value,
            "required_criteria": sorted(REQUIREMENTS[self.measurement]),
            "thresholds": self.thresholds.as_record(),
            "thresholds_are_default": self.thresholds_are_default,
            "screened": len(self.verdicts),
            "admissible": len(self.admissible),
            "needs_frame_check": len(self.needs_frame_check),
            "usable": len(self.usable),
            "rejection_reasons": self.rejection_reasons(),
            "unassessed_criteria": self.unassessed_criteria(),
            "coverage_potential": self.coverage_potential(),
            "admissible_scene_ids": [scene.scene_id for scene in self.admissible],
        }

    def summary(self) -> str:
        lines = [
            f"{self.measurement.value}: {len(self.usable)} / {len(self.verdicts)} usable "
            f"({len(self.admissible)} fully screened, {len(self.needs_frame_check)} pending "
            f"frame checks)"
            f"{'' if self.thresholds_are_default else '   [THRESHOLDS CHANGED]'}",
            f"  requires: {', '.join(sorted(REQUIREMENTS[self.measurement]))}",
        ]
        for reason, count in self.rejection_reasons().items():
            lines.append(f"  rejected {count:4d}  {reason}")
        for reason, count in self.unassessed_criteria().items():
            lines.append(f"  UNASSESSED {count:4d}  {reason} — this catalogue cannot answer it")
        potential = self.coverage_potential()
        lines.append(
            "  coverage potential: "
            + ", ".join(f"{band} {count}" for band, count in potential.items())
        )
        empty = [band for band, count in potential.items() if count == 0]
        if empty:
            lines.append(
                f"  NO MATERIAL for {', '.join(empty)} — a property of the corpus, "
                "not of the screening"
            )
        return "\n".join(lines)


def screen(
    scenes: Sequence[SceneCandidate],
    measurement: Measurement,
    *,
    thresholds: ScreeningThresholds = DEFAULT_THRESHOLDS,
) -> ScreeningReport:
    """Screen a catalogue for one measurement.

    Never widens a threshold to increase yield. A caller supplying different thresholds gets them
    recorded in the report and printed in its summary.
    """
    return ScreeningReport(
        measurement=measurement,
        verdicts=tuple(screen_scene(scene, measurement, thresholds=thresholds) for scene in scenes),
        thresholds=thresholds,
    )


def screen_all(
    scenes: Sequence[SceneCandidate], *, thresholds: ScreeningThresholds = DEFAULT_THRESHOLDS
) -> dict[Measurement, ScreeningReport]:
    """Screen a catalogue for every measurement, since the answers differ."""
    return {
        measurement: screen(scenes, measurement, thresholds=thresholds)
        for measurement in Measurement
    }


__all__ = [
    "CRITERION_NAMES",
    "DEFAULT_THRESHOLDS",
    "REQUIREMENTS",
    "Measurement",
    "SceneCandidate",
    "SceneVerdict",
    "ScreeningReport",
    "ScreeningThresholds",
    "screen",
    "screen_all",
    "screen_scene",
]
