"""Intervals: the temporal unit the catalogue is actually indexed on.

A detected *scene* is the wrong unit. On the traced catalogue, scene detection under-segments so
badly that 21 ranges hold 52.8% of the running time and the longest is 878 seconds — and every
per-scene metric is an aggregate over such a range, so a staticness score computed across 298
seconds says nothing about the five seconds actually sampled from it.

An **interval** is one to three seconds: short enough that its aggregate statistics describe every
frame inside it, which is the only property that makes an aggregate usable for selection.
Intervals overlap, and they are cheap — they are rows over a survey that has already been
computed, so building them costs no decoding at all.

This is also closer to the original intent than detected scenes ever were. The process was always
described as measuring *the same local area over a second or several*; an interval is that span,
and a region is that area.

## What an interval can and cannot say

The survey is downscaled and pre-blurred, so an interval knows about **motion, level, saturation
and cuts** and knows nothing about grain. Two consequences are load-bearing:

* Clipping is reported as how close the *frame extremes* came to the limits, not as a fraction of
  pixels. The fraction needs full-resolution frames and belongs to ``measure.admissibility``.
* Level is decomposed into spread *across* frames and spread *within* frames. A dark interval
  whose ``within_frame_high`` is large contains bright regions — a practical lamp, a window — and
  that is the only way highlight material is going to be found in a film whose frame averages
  never reach the highlight band.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from film_analysis_tools.capabilities.catalogue.survey import FrameSurvey
from film_analysis_tools.core.errors import DataError, SelectionError

DEFAULT_WINDOW_S = 2.0
DEFAULT_STRIDE_S = 1.0

#: Level band edges on the normalised 0..1 code range.
DEFAULT_BAND_EDGES: tuple[float, float] = (0.10, 0.30)
BANDS: tuple[str, ...] = ("shadow", "midtone", "highlight")

#: ``cut_score`` above which a cut is assumed inside the interval. This is ffmpeg's scdet scale,
#: roughly 0-100, and the detector's own default threshold is 10.
DEFAULT_CUT_THRESHOLD = 10.0

EPS = 1.0e-12


@dataclass(frozen=True)
class Interval:
    """A short span of one source, summarised well enough to choose material by."""

    source_id: str
    index: int
    start_s: float
    end_s: float
    sample_count: int

    motion_mean: float
    motion_p90: float
    """The 90th percentile matters more than the mean: one moving frame in an interval is enough
    to contaminate every difference it takes part in."""

    max_cut_score: float

    level_p10: float
    level_p50: float
    level_p90: float
    """Distribution of *frame-mean* level across the interval, normalised 0..1."""

    within_frame_low: float
    within_frame_high: float
    """Mean of the per-frame low and high level percentiles — the spread *inside* a frame.

    A dark interval with a high ``within_frame_high`` contains bright regions. Judging an interval
    by its mean alone is what makes a whole film read as midtone."""

    saturation_mean: float = 0.0
    saturation_p90: float = 0.0
    hue_median: float = 0.0
    bit_depth_min: float = 0.0

    level_slope: float = 0.0
    """Mean change in level per frame. A fade, dissolve or exposure ramp."""

    level_drift: float = 0.0
    """Total level change across the interval."""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def cut_free(self) -> bool:
        return self.max_cut_score <= DEFAULT_CUT_THRESHOLD

    @property
    def band(self) -> str:
        """Band of the interval's median frame level."""
        low, high = DEFAULT_BAND_EDGES
        if self.level_p50 < low:
            return "shadow"
        return "midtone" if self.level_p50 < high else "highlight"

    def contains_band(self, band: str, edges: tuple[float, float] = DEFAULT_BAND_EDGES) -> bool:
        """Whether the interval plausibly contains *regions* in a band.

        Uses the within-frame spread, not the frame mean. This is the difference between asking
        "is this a bright interval" and "does this interval contain anything bright", and only
        the second can find a lamp in a dark room.
        """
        low, high = edges
        if band == "shadow":
            return self.within_frame_low < low
        if band == "midtone":
            return self.within_frame_low < high and self.within_frame_high >= low
        if band == "highlight":
            return self.within_frame_high >= high
        raise SelectionError(f"unknown band {band!r}; expected one of {BANDS}")

    def as_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "index": self.index,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "sample_count": self.sample_count,
            "motion_mean": self.motion_mean,
            "motion_p90": self.motion_p90,
            "max_cut_score": self.max_cut_score,
            "cut_free": self.cut_free,
            "level_p10": self.level_p10,
            "level_p50": self.level_p50,
            "level_p90": self.level_p90,
            "within_frame_low": self.within_frame_low,
            "within_frame_high": self.within_frame_high,
            "band": self.band,
            "saturation_mean": self.saturation_mean,
            "saturation_p90": self.saturation_p90,
            "hue_median": self.hue_median,
            "bit_depth_min": self.bit_depth_min,
            "level_slope": self.level_slope,
            "level_drift": self.level_drift,
        }


def build_intervals(
    survey: FrameSurvey,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    stride_s: float = DEFAULT_STRIDE_S,
    drop_cuts: bool = True,
) -> list[Interval]:
    """Aggregate a survey into overlapping intervals.

    Costs no decoding: the survey has already been computed. ``drop_cuts`` discards intervals
    containing a detected cut, since a difference across a cut is two pictures subtracted rather
    than a residual.
    """
    if window_s <= 0 or stride_s <= 0:
        raise SelectionError(f"window and stride must be positive: {window_s}, {stride_s}")

    per_window = max(2, round(window_s * survey.sample_rate_hz))
    step = max(1, round(stride_s * survey.sample_rate_hz))
    total = len(survey)
    if total < per_window:
        raise DataError(
            f"survey has {total} samples, fewer than the {per_window} a {window_s}s interval needs"
        )

    sample_period = 1.0 / survey.sample_rate_hz
    time = survey.column("time_s")
    motion = survey.column("motion")
    cut = survey.column("cut_score")
    level = survey.normalised("level_mean")
    low = survey.normalised("level_low") if survey.has("level_low") else level
    high = survey.normalised("level_high") if survey.has("level_high") else level
    saturation = (
        survey.column("saturation_mean") if survey.has("saturation_mean") else np.zeros(total)
    )
    hue = survey.column("hue_median") if survey.has("hue_median") else np.zeros(total)
    depth = survey.column("bit_depth") if survey.has("bit_depth") else np.zeros(total)

    intervals: list[Interval] = []
    for index, start in enumerate(range(0, total - per_window + 1, step)):
        window = slice(start, start + per_window)
        max_cut = float(cut[window].max())
        if drop_cuts and max_cut > DEFAULT_CUT_THRESHOLD:
            continue

        window_levels = level[window]
        slope = (
            float(np.polyfit(np.arange(per_window, dtype=np.float64), window_levels, 1)[0])
            if per_window >= 2
            else 0.0
        )
        intervals.append(
            Interval(
                source_id=survey.source_id,
                index=index,
                start_s=float(time[start]),
                # Half-open: [start_s, end_s). The span ends one sample period after the last
                # sample, not *at* it, so an interval requested as 2.0 s at 4 Hz covers 8 samples
                # and reports 2.0 s. Taking the last sample's timestamp reported 1.75 s, which
                # made `min_duration_s=2.0` reject a window asked for as two seconds and lost one
                # sample period of coverage from every span.
                end_s=float(time[start + per_window - 1]) + sample_period,
                sample_count=per_window,
                motion_mean=float(motion[window].mean()),
                motion_p90=float(np.percentile(motion[window], 90)),
                max_cut_score=max_cut,
                level_p10=float(np.percentile(window_levels, 10)),
                level_p50=float(np.percentile(window_levels, 50)),
                level_p90=float(np.percentile(window_levels, 90)),
                within_frame_low=float(low[window].mean()),
                within_frame_high=float(high[window].mean()),
                saturation_mean=float(saturation[window].mean()),
                saturation_p90=float(np.percentile(saturation[window], 90)),
                hue_median=float(np.median(hue[window])),
                bit_depth_min=float(depth[window].min()),
                level_slope=slope,
                level_drift=float(window_levels[-1] - window_levels[0]),
            )
        )
    return intervals


@dataclass(frozen=True)
class IntervalIndex:
    """A queryable set of intervals over one or more sources."""

    intervals: tuple[Interval, ...]

    def __len__(self) -> int:
        return len(self.intervals)

    def __iter__(self) -> Any:
        return iter(self.intervals)

    def select(
        self,
        *,
        max_motion: float | None = None,
        min_duration_s: float | None = None,
        band: str | None = None,
        contains_band: str | None = None,
        min_saturation: float | None = None,
        max_level_drift: float | None = None,
        min_bit_depth: float | None = None,
        cut_free: bool = True,
        source_id: str | None = None,
        per_source_cap: int = 0,
        limit: int = 0,
    ) -> list[Interval]:
        """Intervals matching every stated condition.

        ``band`` asks what the interval *is*; ``contains_band`` asks what it *contains*. The
        second is what finds a lamp in a dark room, and the first cannot.
        """
        chosen: list[Interval] = []
        seen: dict[str, int] = {}
        for interval in self.intervals:
            if cut_free and not interval.cut_free:
                continue
            if source_id is not None and interval.source_id != source_id:
                continue
            if max_motion is not None and interval.motion_p90 > max_motion:
                continue
            if min_duration_s is not None and interval.duration_s < min_duration_s:
                continue
            if band is not None and interval.band != band:
                continue
            if contains_band is not None and not interval.contains_band(contains_band):
                continue
            if min_saturation is not None and interval.saturation_p90 < min_saturation:
                continue
            if max_level_drift is not None and abs(interval.level_drift) > max_level_drift:
                continue
            if min_bit_depth is not None and interval.bit_depth_min < min_bit_depth:
                continue
            if per_source_cap:
                used = seen.get(interval.source_id, 0)
                if used >= per_source_cap:
                    continue
                seen[interval.source_id] = used + 1
            chosen.append(interval)
            if limit and len(chosen) >= limit:
                break
        return chosen

    def band_counts(self, *, contains: bool = False) -> dict[str, int]:
        """Band populations, either by what intervals *are* or by what they *contain*."""
        if contains:
            return {band: sum(1 for i in self.intervals if i.contains_band(band)) for band in BANDS}
        counts = dict.fromkeys(BANDS, 0)
        for interval in self.intervals:
            counts[interval.band] += 1
        return counts

    def total_duration_s(self) -> float:
        """Union of covered time, so overlapping intervals are not counted twice.

        Unioned **within each source and then summed**. Different sources have independent time
        axes: two films each contributing 0-2 s cover four seconds of material, not two. Pooling
        the spans made them collide at the origin and under-reported coverage by exactly the
        overlap between unrelated timelines.
        """
        total = 0.0
        for source in {interval.source_id for interval in self.intervals}:
            spans = sorted((i.start_s, i.end_s) for i in self.intervals if i.source_id == source)
            current_end = -np.inf
            for start, end in spans:
                if start > current_end:
                    total += end - start
                    current_end = end
                elif end > current_end:
                    total += end - current_end
                    current_end = end
        return float(total)

    def as_record(self) -> dict[str, Any]:
        return {
            "intervals": len(self.intervals),
            "sources": sorted({i.source_id for i in self.intervals}),
            "covered_s": self.total_duration_s(),
            "band_counts": self.band_counts(),
            "contains_band_counts": self.band_counts(contains=True),
        }

    def summary(self) -> str:
        record = self.as_record()
        lines = [
            f"{len(self)} intervals over {len(record['sources'])} source(s), "
            f"covering {record['covered_s'] / 60:.1f} min",
            "  interval band (what it is)      : "
            + ", ".join(f"{k} {v}" for k, v in record["band_counts"].items()),
            "  contains band (what it holds)   : "
            + ", ".join(f"{k} {v}" for k, v in record["contains_band_counts"].items()),
        ]
        return "\n".join(lines)


def index(intervals: Sequence[Interval]) -> IntervalIndex:
    return IntervalIndex(intervals=tuple(intervals))


__all__ = [
    "BANDS",
    "DEFAULT_BAND_EDGES",
    "DEFAULT_CUT_THRESHOLD",
    "DEFAULT_STRIDE_S",
    "DEFAULT_WINDOW_S",
    "Interval",
    "IntervalIndex",
    "build_intervals",
    "index",
]
