"""Attaching face and colour evidence to intervals, with its evidential status attached too.

Step 2 of the architecture: make face-bearing and colour-conditioned intervals *queryable* rather
than inferred by reading scripts.

The two annotations are not equally trustworthy, and the types say so.

**Colour is measured for the interval.** The survey sampled every frame, so an interval's
saturation and hue come from its own frames.

**Face presence is inherited.** The face scout probed **one frame per scene, at the scene
midpoint**. With a median scene of 6.6 s that probe is within ~3 s of everything in it, but the
90th percentile is 49 s and the longest scene is 878 s — so an interval can sit **439 seconds**
from the only frame that was ever checked for a face. Only 54 of 216 face-bearing scenes are short
enough for the probe to be within 1.5 s of every interval.

So a face annotation carries the distance to the observation that produced it, and a confidence
tier derived from that distance. An interval 400 seconds from the probe is not evidence that a
face is present; it is evidence that a face was present *somewhere in the same detected range*,
which — given how badly those ranges are segmented — is close to no evidence at all.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from film_analysis_tools.capabilities.catalogue.intervals import Interval
from film_analysis_tools.core.errors import SelectionError

# Saturation band edges on the signalstats scale, taken from the measured distribution across
# 37,084 frames: p25 13.2, p75 21.4, p95 32.2. Chosen after looking, not before.
SATURATION_EDGES: tuple[float, float, float] = (12.0, 22.0, 32.0)
SATURATION_BANDS: tuple[str, ...] = ("neutral", "low", "moderate", "saturated")

#: A probe inside the interval is an observation. Beyond this it is inherited from elsewhere.
NEAR_PROBE_S = 2.0
#: Beyond this the scene-level inheritance is too weak to act on.
USABLE_PROBE_S = 10.0


class FaceConfidence(Enum):
    """How far the face evidence is from the interval it is attached to."""

    OBSERVED = "observed"
    """The probe frame lies inside the interval."""

    NEAR = "near"
    """The probe is within a couple of seconds."""

    INHERITED = "inherited"
    """The probe is in the same detected range but far away. Weak."""

    NONE = "none"
    """No face observation applies."""

    @property
    def usable_for_skin(self) -> bool:
        """Whether a skin measurement may rest on this.

        Deliberately excludes ``INHERITED``: measuring skin grain on an interval whose only
        evidence of a face is a frame minutes away would be measuring whatever happens to be
        there instead.
        """
        return self in (FaceConfidence.OBSERVED, FaceConfidence.NEAR)


@dataclass(frozen=True)
class FaceObservation:
    """A single probe frame that was checked for faces."""

    time_s: float
    detected: bool
    count: int = 0
    area_ratio: float = 0.0
    detection_score: float = 0.0
    bbox: tuple[float, float, float, float] | None = None
    source_scene: str = ""
    source_id: str = ""
    """Which source this probe was taken from.

    Empty means unscoped. A timestamp alone is not an address: two films both have a second 1.0,
    and joining on time alone let a probe from one film mark intervals in another as carrying an
    *observed* face. :func:`annotate` accepts unscoped probes only when there is exactly one
    source to attach them to, and refuses to guess otherwise.
    """


@dataclass(frozen=True)
class FaceAnnotation:
    """Face evidence for one interval, and how far away it was measured."""

    observation: FaceObservation | None
    distance_s: float

    @property
    def detected(self) -> bool:
        return bool(self.observation and self.observation.detected)

    @property
    def confidence(self) -> FaceConfidence:
        if self.observation is None or not self.observation.detected:
            return FaceConfidence.NONE
        if self.distance_s <= 0.0:
            return FaceConfidence.OBSERVED
        if self.distance_s <= NEAR_PROBE_S:
            return FaceConfidence.NEAR
        return FaceConfidence.INHERITED

    @property
    def area_ratio(self) -> float:
        return self.observation.area_ratio if self.observation else 0.0

    def as_record(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "confidence": self.confidence.value,
            "distance_s": self.distance_s,
            "area_ratio": self.area_ratio,
            "count": self.observation.count if self.observation else 0,
            "detection_score": self.observation.detection_score if self.observation else 0.0,
            "bbox": list(self.observation.bbox)
            if self.observation and self.observation.bbox
            else None,
            "source_scene": self.observation.source_scene if self.observation else "",
        }


@dataclass(frozen=True)
class ColourAnnotation:
    """Colour evidence for one interval. Measured from the interval's own frames."""

    saturation_band: str
    saturation_p90: float
    dominant_hue_deg: float

    @property
    def is_neutral(self) -> bool:
        return self.saturation_band == "neutral"

    @property
    def is_saturated(self) -> bool:
        return self.saturation_band == "saturated"

    def as_record(self) -> dict[str, Any]:
        return {
            "saturation_band": self.saturation_band,
            "saturation_p90": self.saturation_p90,
            "dominant_hue_deg": self.dominant_hue_deg,
        }


def saturation_band(value: float) -> str:
    low, moderate, high = SATURATION_EDGES
    if value < low:
        return "neutral"
    if value < moderate:
        return "low"
    return "moderate" if value < high else "saturated"


def colour_annotation(interval: Interval) -> ColourAnnotation:
    """Colour for an interval, from its own frames.

    Saturation uses the interval's **90th percentile**, not its mean: the question worth asking
    is whether the interval *contains* saturated content, and a mean over a mostly-neutral frame
    hides a single vivid object. Hue is the frame *median*, which describes a dominant cast and
    cannot say what a frame contains — it is recorded for grouping, not for selection by content.
    """
    return ColourAnnotation(
        saturation_band=saturation_band(interval.saturation_p90),
        saturation_p90=interval.saturation_p90,
        dominant_hue_deg=interval.hue_median,
    )


@dataclass(frozen=True)
class AnnotatedInterval:
    """An interval with its face and colour evidence attached."""

    interval: Interval
    face: FaceAnnotation
    colour: ColourAnnotation

    @property
    def source_id(self) -> str:
        return self.interval.source_id

    @property
    def start_s(self) -> float:
        return self.interval.start_s

    @property
    def duration_s(self) -> float:
        return self.interval.duration_s

    def as_record(self) -> dict[str, Any]:
        return {
            **self.interval.as_record(),
            "face": self.face.as_record(),
            "colour": self.colour.as_record(),
        }


def annotate(
    intervals: Sequence[Interval], observations: Sequence[FaceObservation] = ()
) -> list[AnnotatedInterval]:
    """Attach colour and the nearest face observation to each interval.

    The join is temporal *within a source*: an interval takes the closest probe in time from its
    own source, and records how far away it was. Nothing is dropped for being distant — the
    distance is reported and the caller decides what it is worth.

    Sources have independent time axes, so a probe only ever joins intervals from the source it
    came from. An observation with no ``source_id`` is unscoped; it is accepted only when the
    intervals name exactly one source, and raises otherwise rather than attaching a face from one
    film to another.
    """
    interval_sources = {interval.source_id for interval in intervals}
    unscoped = [o for o in observations if not o.source_id]
    if unscoped and len(interval_sources) > 1:
        raise SelectionError(
            f"{len(unscoped)} face observation(s) carry no source_id but the intervals span "
            f"{len(interval_sources)} sources ({sorted(interval_sources)}). A timestamp alone "
            "cannot say which source a probe belongs to; set FaceObservation.source_id."
        )

    only_source = next(iter(interval_sources)) if len(interval_sources) == 1 else ""
    by_source: dict[str, list[FaceObservation]] = {}
    for observation in observations:
        key = observation.source_id or only_source
        by_source.setdefault(key, []).append(observation)
    ordered_by_source = {
        key: sorted(group, key=lambda observation: observation.time_s)
        for key, group in by_source.items()
    }
    times_by_source = {
        key: [observation.time_s for observation in group]
        for key, group in ordered_by_source.items()
    }

    annotated: list[AnnotatedInterval] = []
    for interval in intervals:
        ordered = ordered_by_source.get(interval.source_id, [])
        times = times_by_source.get(interval.source_id, [])
        nearest, distance = None, float("inf")
        if ordered:
            position = bisect.bisect_left(times, interval.start_s)
            for candidate in ordered[max(0, position - 1) : position + 2]:
                if interval.start_s <= candidate.time_s < interval.end_s:
                    gap = 0.0
                elif candidate.time_s < interval.start_s:
                    gap = interval.start_s - candidate.time_s
                else:
                    gap = candidate.time_s - interval.end_s
                if gap < distance:
                    nearest, distance = candidate, gap
        annotated.append(
            AnnotatedInterval(
                interval=interval,
                face=FaceAnnotation(
                    observation=nearest, distance_s=0.0 if nearest is None else distance
                ),
                colour=colour_annotation(interval),
            )
        )
    return annotated


@dataclass(frozen=True)
class AnnotatedIndex:
    """A queryable set of annotated intervals — the catalogue the extractor asks."""

    intervals: tuple[AnnotatedInterval, ...]

    def __len__(self) -> int:
        return len(self.intervals)

    def __iter__(self) -> Any:
        return iter(self.intervals)

    def select(
        self,
        *,
        max_motion: float | None = None,
        min_duration_s: float | None = None,
        contains_band: str | None = None,
        band: str | None = None,
        has_face: bool | None = None,
        face_confidence: FaceConfidence | None = None,
        min_face_area: float | None = None,
        saturation: str | None = None,
        cut_free: bool = True,
        per_source_cap: int = 0,
        limit: int = 0,
    ) -> list[AnnotatedInterval]:
        """Intervals matching every stated condition.

        ``face_confidence`` filters on how the face evidence was obtained, not merely on whether
        a face was reported. Asking for skin material without stating it is how an interval
        minutes from the only checked frame ends up in a skin corpus.
        """
        if saturation is not None and saturation not in SATURATION_BANDS:
            raise SelectionError(
                f"unknown saturation band {saturation!r}; expected one of {SATURATION_BANDS}"
            )
        chosen: list[AnnotatedInterval] = []
        seen: dict[str, int] = {}
        for entry in self.intervals:
            base = entry.interval
            if cut_free and not base.cut_free:
                continue
            if max_motion is not None and base.motion_p90 > max_motion:
                continue
            if min_duration_s is not None and base.duration_s < min_duration_s:
                continue
            if band is not None and base.band != band:
                continue
            if contains_band is not None and not base.contains_band(contains_band):
                continue
            if has_face is not None and entry.face.detected != has_face:
                continue
            if face_confidence is not None and entry.face.confidence is not face_confidence:
                continue
            if min_face_area is not None and entry.face.area_ratio < min_face_area:
                continue
            if saturation is not None and entry.colour.saturation_band != saturation:
                continue
            if per_source_cap:
                used = seen.get(entry.source_id, 0)
                if used >= per_source_cap:
                    continue
                seen[entry.source_id] = used + 1
            chosen.append(entry)
            if limit and len(chosen) >= limit:
                break
        return chosen

    def face_confidence_counts(self) -> dict[str, int]:
        counts = dict.fromkeys((tier.value for tier in FaceConfidence), 0)
        for entry in self.intervals:
            counts[entry.face.confidence.value] += 1
        return counts

    def saturation_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(SATURATION_BANDS, 0)
        for entry in self.intervals:
            counts[entry.colour.saturation_band] += 1
        return counts

    def as_record(self) -> dict[str, Any]:
        return {
            "intervals": len(self.intervals),
            "face_confidence": self.face_confidence_counts(),
            "saturation": self.saturation_counts(),
        }

    def summary(self) -> str:
        face = self.face_confidence_counts()
        usable = sum(count for tier, count in face.items() if FaceConfidence(tier).usable_for_skin)
        return "\n".join(
            [
                f"{len(self)} annotated intervals",
                "  face evidence : " + ", ".join(f"{k} {v}" for k, v in face.items() if v),
                f"                  {usable} usable for skin "
                "(observed or near; inherited evidence is not acted on)",
                "  saturation    : "
                + ", ".join(f"{k} {v}" for k, v in self.saturation_counts().items() if v),
            ]
        )


def index(intervals: Sequence[AnnotatedInterval]) -> AnnotatedIndex:
    return AnnotatedIndex(intervals=tuple(intervals))


def observations_from_probes(
    times_s: Sequence[float],
    detected: Sequence[bool],
    *,
    counts: Sequence[int] | None = None,
    area_ratios: Sequence[float] | None = None,
    scores: Sequence[float] | None = None,
    scenes: Sequence[str] | None = None,
    source_id: str = "",
) -> list[FaceObservation]:
    """Build observations from parallel columns, whatever produced them."""
    length = len(times_s)
    if len(detected) != length:
        raise SelectionError("times and detections must be the same length")
    zeros_i = counts or [0] * length
    zeros_f = area_ratios or [0.0] * length
    scores_f = scores or [0.0] * length
    names = scenes or [""] * length
    return [
        FaceObservation(
            time_s=float(times_s[i]),
            detected=bool(detected[i]),
            count=int(zeros_i[i]),
            area_ratio=float(zeros_f[i]),
            detection_score=float(scores_f[i]),
            source_scene=str(names[i]),
            source_id=source_id,
        )
        for i in range(length)
    ]


__all__ = [
    "NEAR_PROBE_S",
    "SATURATION_BANDS",
    "SATURATION_EDGES",
    "USABLE_PROBE_S",
    "AnnotatedIndex",
    "AnnotatedInterval",
    "ColourAnnotation",
    "FaceAnnotation",
    "FaceConfidence",
    "FaceObservation",
    "annotate",
    "colour_annotation",
    "index",
    "observations_from_probes",
    "saturation_band",
]
