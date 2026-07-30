"""The region record: a measured tile that knows where it came from.

Step 3 of the architecture. :class:`~film_analysis_tools.capabilities.measure.windows.Window`
already carries everything a region needs to be measured — position, size, level, motion energy,
structure, band, texture — and then it is discarded at the end of the run. A window has no
identity: it cannot say which source it came from, which interval, which frames, or under what
gate it was admitted. So tiles are recomputed inside every analysis, never compared across runs,
and a fitted curve cannot name the material behind any of its knots.

A :class:`Region` is a window plus that provenance. Three things it records are not decoration:

**The level scale.** A level of 0.18 means one thing on a linear scale and something entirely
different in PQ code values. Regions measured on different scales must never land in the same
amplitude curve, and without the scale recorded nothing would notice: the fit would simply be
wrong in a way that looks fine.

**The gate, and whether it was the default one.** Widening the motion gate mines moving
background. :attr:`RegionIndex.select` therefore excludes widened-gate regions *by default* —
they must be asked for, which is the opposite of the legacy behaviour where a widened run was
indistinguishable afterwards.

**The interval it belongs to.** This is what makes independence countable. Tiling a single
interval into 500 windows yields 500 rows and nowhere near 500 independent samples, and intervals
built with overlapping windows share frames with their neighbours. :meth:`RegionIndex.independence`
reports the merged, disjoint time actually behind a set of regions, so a corpus cannot claim a
sample size it does not have.

Colour and face membership are typed as *absent* rather than zero until something measures them,
following :mod:`film_analysis_tools.capabilities.measure.chroma`: a region tiled from a luma stack
has no colour to report, and saying ``0.0`` would be a claim.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from film_analysis_tools.capabilities.measure.windows import (
    BANDS,
    DEFAULT_BAND_EDGES,
    DEFAULT_GATE,
    POSITIONS,
    TEXTURES,
    SelectionReport,
    Window,
    WindowGate,
)
from film_analysis_tools.core.errors import SelectionError

#: Level scales a region may be measured on. A region must declare one; there is no default,
#: because every plausible default is silently wrong for some source.
LEVEL_SCALES: tuple[str, ...] = ("linear", "pq_code", "hlg_code", "slog3", "gamma_code")


@dataclass(frozen=True)
class RegionColour:
    """Colour of a region, for *selection* only.

    This says what colour the region is, so material can be chosen by it. It says nothing about
    how grain couples between emulsion layers — that comes from the spectral model, never from a
    delivery master (architecture §1).
    """

    saturation: float
    hue_deg: float

    def as_record(self) -> dict[str, Any]:
        return {"saturation": self.saturation, "hue_deg": self.hue_deg}


@dataclass(frozen=True)
class RegionFace:
    """How much of the region is skin, from facial geometry."""

    mesh_fraction: float
    """Fraction of the tile inside the face mesh. 1.0 is entirely skin."""

    def as_record(self) -> dict[str, Any]:
        return {"mesh_fraction": self.mesh_fraction}


@dataclass(frozen=True)
class RegionProvenance:
    """Where a region came from and what admitted it."""

    source_id: str
    interval_start_s: float
    interval_end_s: float
    start_frame: int
    frames: int
    level_scale: str
    gate: WindowGate = DEFAULT_GATE
    band_edges: tuple[float, float] = DEFAULT_BAND_EDGES
    source_identity: str = ""
    """:attr:`SourceRecord.identity` — empty when the region is not bound to a source record."""

    def __post_init__(self) -> None:
        if self.level_scale not in LEVEL_SCALES:
            raise SelectionError(
                f"unknown level scale {self.level_scale!r}; expected one of {LEVEL_SCALES}. "
                "A level without a scale cannot be compared with another level."
            )
        if self.interval_end_s < self.interval_start_s:
            raise SelectionError(
                f"interval ends before it starts: {self.interval_start_s}..{self.interval_end_s}"
            )
        if self.frames <= 0:
            raise SelectionError(f"a region must span at least one frame; got {self.frames}")

    @property
    def gate_is_default(self) -> bool:
        return self.gate == DEFAULT_GATE

    @property
    def interval_key(self) -> str:
        """Identifies the interval, so regions sharing one can be counted as what they are."""
        return f"{self.source_id}@{self.interval_start_s:.3f}"

    def as_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_identity": self.source_identity,
            "interval_start_s": self.interval_start_s,
            "interval_end_s": self.interval_end_s,
            "start_frame": self.start_frame,
            "frames": self.frames,
            "level_scale": self.level_scale,
            "gate": self.gate.as_record(),
            "gate_is_default": self.gate_is_default,
            "band_edges": list(self.band_edges),
        }


@dataclass(frozen=True)
class Region:
    """A measured tile, addressable and traceable."""

    window: Window
    provenance: RegionProvenance
    colour: RegionColour | None = None
    face: RegionFace | None = None

    @property
    def region_id(self) -> str:
        """A readable address: which source, which frames, which rectangle."""
        provenance, window = self.provenance, self.window
        return (
            f"{provenance.source_id}@{provenance.start_frame}+{provenance.frames}"
            f":{window.x},{window.y}+{window.size}"
        )

    # -- the window's measurements, reachable without unwrapping ------------------
    @property
    def x(self) -> int:
        return self.window.x

    @property
    def y(self) -> int:
        return self.window.y

    @property
    def size(self) -> int:
        return self.window.size

    @property
    def level(self) -> float:
        return self.window.level

    @property
    def band(self) -> str:
        return self.window.band

    @property
    def texture(self) -> str:
        return self.window.texture

    @property
    def position(self) -> str:
        return self.window.position

    @property
    def motion_energy(self) -> float:
        return self.window.motion_energy

    @property
    def stratum(self) -> tuple[str, str, str]:
        return self.window.stratum

    @property
    def level_low(self) -> float:
        return self.window.level_low

    @property
    def level_high(self) -> float:
        return self.window.level_high

    def contains_band(self, band: str) -> bool:
        """What the tile *holds*, against the band edges it was measured with."""
        return self.window.contains_band(band, self.provenance.band_edges)

    # -- provenance shortcuts ----------------------------------------------------
    @property
    def source_id(self) -> str:
        return self.provenance.source_id

    @property
    def interval_key(self) -> str:
        return self.provenance.interval_key

    @property
    def level_scale(self) -> str:
        return self.provenance.level_scale

    def as_record(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            **self.window.as_record(),
            "provenance": self.provenance.as_record(),
            "colour": self.colour.as_record() if self.colour else None,
            "face": self.face.as_record() if self.face else None,
        }


def regions_from_report(
    report: SelectionReport,
    *,
    source_id: str,
    interval_start_s: float,
    interval_end_s: float,
    start_frame: int,
    level_scale: str,
    source_identity: str = "",
) -> list[Region]:
    """Turn accepted windows into indexed regions — the step that stops discarding the work.

    Only accepted windows become regions. Rejections stay in the report: they describe why this
    material was not usable, which is a property of the run, not a catalogue entry.
    """
    provenance = RegionProvenance(
        source_id=source_id,
        interval_start_s=interval_start_s,
        interval_end_s=interval_end_s,
        start_frame=start_frame,
        frames=report.frames,
        level_scale=level_scale,
        gate=report.gate,
        band_edges=report.band_edges,
        source_identity=source_identity,
    )
    return [Region(window=window, provenance=provenance) for window in report.accepted]


def merged_span_seconds(regions: Sequence[Region]) -> tuple[int, float]:
    """Disjoint time actually behind a set of regions: ``(spans, seconds)``.

    Intervals are built overlapping — 2 s windows on a 1 s stride — so neighbouring intervals
    share half their frames. Counting them as separate samples double-counts the material.
    """
    spans = sorted({(r.provenance.interval_start_s, r.provenance.interval_end_s) for r in regions})
    if not spans:
        return (0, 0.0)
    merged: list[list[float]] = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return (len(merged), sum(end - start for start, end in merged))


@dataclass(frozen=True)
class Independence:
    """How many independent samples a set of regions really represents.

    A region count is not a sample size. Five hundred tiles from one interval are five hundred
    correlated views of the same few seconds of film.
    """

    regions: int
    intervals: int
    sources: int
    spans: int
    span_seconds: float
    max_regions_per_interval: int

    @property
    def regions_per_span(self) -> float:
        return self.regions / self.spans if self.spans else 0.0

    @property
    def concentrated(self) -> bool:
        """True when a single interval supplies more than a quarter of the regions."""
        return bool(self.regions) and self.max_regions_per_interval > self.regions / 4

    def as_record(self) -> dict[str, Any]:
        return {
            "regions": self.regions,
            "intervals": self.intervals,
            "sources": self.sources,
            "spans": self.spans,
            "span_seconds": self.span_seconds,
            "max_regions_per_interval": self.max_regions_per_interval,
            "regions_per_span": self.regions_per_span,
            "concentrated": self.concentrated,
        }

    def summary(self) -> str:
        line = (
            f"{self.regions:,} regions from {self.intervals:,} intervals "
            f"({self.spans:,} disjoint spans, {self.span_seconds:.1f}s) in "
            f"{self.sources} source(s)"
        )
        note = (
            f"  treat ~{self.spans:,} as the independent count, not {self.regions:,}: "
            f"regions tile the same frames ({self.regions_per_span:.0f} per span)"
        )
        if self.concentrated:
            note += (
                f"\n  CONCENTRATED: one interval supplies {self.max_regions_per_interval:,} of them"
            )
        return f"{line}\n{note}"


@dataclass(frozen=True)
class RegionIndex:
    """A queryable set of regions — what a corpus is assembled from."""

    regions: tuple[Region, ...]

    def __len__(self) -> int:
        return len(self.regions)

    def __iter__(self) -> Iterator[Region]:
        return iter(self.regions)

    def select(
        self,
        *,
        band: str | None = None,
        contains_band: str | None = None,
        texture: str | None = None,
        position: str | None = None,
        min_level: float | None = None,
        max_level: float | None = None,
        max_motion_energy: float | None = None,
        source: str | None = None,
        level_scale: str | None = None,
        require_default_gate: bool = True,
        require_colour: bool = False,
        require_face: bool = False,
        min_face_fraction: float | None = None,
        per_interval_cap: int = 0,
        per_source_cap: int = 0,
        limit: int = 0,
    ) -> list[Region]:
        """Regions matching every stated condition.

        ``require_default_gate`` defaults to **true**: a region admitted under a widened gate is
        excluded unless explicitly asked for. Yield is never a reason to relax a gate, and a
        corpus should not acquire widened-gate material by not mentioning it.

        ``per_interval_cap`` is the practical defence against the concentration
        :class:`Independence` reports — it forces spread across material rather than depth within
        one interval.
        """
        for name, value, allowed in (
            ("band", band, BANDS),
            ("band", contains_band, BANDS),
            ("texture", texture, TEXTURES),
            ("position", position, POSITIONS),
            ("level scale", level_scale, LEVEL_SCALES),
        ):
            if value is not None and value not in allowed:
                raise SelectionError(f"unknown {name} {value!r}; expected one of {allowed}")

        chosen: list[Region] = []
        per_interval: dict[str, int] = {}
        per_source: dict[str, int] = {}
        for region in self.regions:
            if require_default_gate and not region.provenance.gate_is_default:
                continue
            if band is not None and region.band != band:
                continue
            if contains_band is not None and not region.contains_band(contains_band):
                continue
            if texture is not None and region.texture != texture:
                continue
            if position is not None and region.position != position:
                continue
            if min_level is not None and region.level < min_level:
                continue
            if max_level is not None and region.level > max_level:
                continue
            if max_motion_energy is not None and region.motion_energy > max_motion_energy:
                continue
            if source is not None and region.source_id != source:
                continue
            if level_scale is not None and region.level_scale != level_scale:
                continue
            if require_colour and region.colour is None:
                continue
            if (require_face or min_face_fraction is not None) and region.face is None:
                continue
            if (
                min_face_fraction is not None
                and region.face is not None
                and region.face.mesh_fraction < min_face_fraction
            ):
                continue
            if per_interval_cap:
                used = per_interval.get(region.interval_key, 0)
                if used >= per_interval_cap:
                    continue
                per_interval[region.interval_key] = used + 1
            if per_source_cap:
                used = per_source.get(region.source_id, 0)
                if used >= per_source_cap:
                    continue
                per_source[region.source_id] = used + 1
            chosen.append(region)
            if limit and len(chosen) >= limit:
                break
        return chosen

    def coverage(self, *, contains: bool = False) -> dict[str, int]:
        """Band counts. ``contains=True`` asks what tiles *hold* rather than what they average to.

        The two differ, and the difference is not small: on the 4K master 18 of 120 tiles in one
        interval reached the highlight edge by mean and 32 by content.
        """
        counts = dict.fromkeys(BANDS, 0)
        for region in self.regions:
            if contains:
                for name in BANDS:
                    counts[name] += int(region.contains_band(name))
            else:
                counts[region.band] += 1
        return counts

    def texture_coverage(self) -> dict[str, int]:
        counts = dict.fromkeys(TEXTURES, 0)
        for region in self.regions:
            counts[region.texture] += 1
        return counts

    def level_range(self) -> tuple[float, float]:
        if not self.regions:
            return (0.0, 0.0)
        levels = [region.level for region in self.regions]
        return (min(levels), max(levels))

    def level_scales(self) -> list[str]:
        """More than one means these regions must not be pooled into a single curve."""
        return sorted({region.level_scale for region in self.regions})

    def band_edges(self) -> list[tuple[float, float]]:
        return sorted({region.provenance.band_edges for region in self.regions})

    @property
    def edges_bracket_the_data(self) -> bool:
        """Whether the band edges fall inside the measured level range.

        When they do not, every region lands in one band *by construction* and coverage collapses.
        That looks identical to "this material has no highlights", but the two need opposite
        fixes — shoot different material, or correct the units — so they must be told apart.

        Measured on the 4K PQ master: linear levels normalised so 1.0 is the PQ peak span
        0.00000-0.02865, while the default edges are (0.02, 0.25). 737 of 739 regions were
        labelled shadow. The film is not that dark; diffuse white simply sits near 0.01 on a scale
        whose 1.0 means 10,000 nits.
        """
        if not self.regions:
            return True
        low, high = self.level_range()
        return any(low <= edge <= high for edges in self.band_edges() for edge in edges)

    def independence(self) -> Independence:
        per_interval: dict[str, int] = {}
        for region in self.regions:
            per_interval[region.interval_key] = per_interval.get(region.interval_key, 0) + 1
        spans, seconds = merged_span_seconds(self.regions)
        return Independence(
            regions=len(self.regions),
            intervals=len(per_interval),
            sources=len({region.source_id for region in self.regions}),
            spans=spans,
            span_seconds=seconds,
            max_regions_per_interval=max(per_interval.values(), default=0),
        )

    def widened_gate_count(self) -> int:
        return sum(1 for region in self.regions if not region.provenance.gate_is_default)

    def as_record(self) -> dict[str, Any]:
        low, high = self.level_range()
        return {
            "regions": len(self.regions),
            "coverage": self.coverage(),
            "contains_band_coverage": self.coverage(contains=True),
            "texture_coverage": self.texture_coverage(),
            "level_min": low,
            "level_max": high,
            "level_scales": self.level_scales(),
            "band_edges": [list(edges) for edges in self.band_edges()],
            "edges_bracket_the_data": self.edges_bracket_the_data,
            "widened_gate": self.widened_gate_count(),
            "independence": self.independence().as_record(),
        }

    def summary(self) -> str:
        low, high = self.level_range()
        scales = self.level_scales()
        lines = [
            self.independence().summary(),
            f"  levels : {low:.5f} to {high:.5f} on {'/'.join(scales) or 'no'} scale",
            "  bands  : " + ", ".join(f"{k} {v}" for k, v in self.coverage().items()) + "  (is)",
            "  bands  : "
            + ", ".join(f"{k} {v}" for k, v in self.coverage(contains=True).items())
            + "  (contains)",
            "  texture: " + ", ".join(f"{k} {v}" for k, v in self.texture_coverage().items()),
        ]
        if len(scales) > 1:
            lines.append(
                f"  MIXED SCALES {scales}: these levels are not comparable and must not be "
                "pooled into one amplitude curve."
            )
        if self.regions and not self.edges_bracket_the_data:
            edges = ", ".join(str(list(pair)) for pair in self.band_edges())
            lines.append(
                f"  EDGES DO NOT BRACKET THE DATA: levels {low:.5f}..{high:.5f} against edges "
                f"{edges}. Every region falls in one band by construction, so coverage here "
                "means nothing. This is a units mismatch, not an absence of material — check "
                "what 1.0 means on the '" + (scales[0] if scales else "?") + "' scale before "
                "concluding the footage is unusable."
            )
        widened = self.widened_gate_count()
        if widened:
            lines.append(
                f"  {widened:,} region(s) came from a widened gate and are excluded from "
                "select() unless require_default_gate=False."
            )
        return "\n".join(lines)


def index(regions: Sequence[Region]) -> RegionIndex:
    return RegionIndex(regions=tuple(regions))


__all__ = [
    "LEVEL_SCALES",
    "Independence",
    "Region",
    "RegionColour",
    "RegionFace",
    "RegionIndex",
    "RegionProvenance",
    "index",
    "merged_span_seconds",
    "regions_from_report",
]
