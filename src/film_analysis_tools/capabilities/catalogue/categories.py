"""The validation taxonomy: what each category is for, and how membership is decided.

These categories exist to stress a colour module in the ways it actually breaks — collapsed
colour, hard clipping, unstable gradients, noise amplification — so each one names the failure
it is meant to provoke rather than merely describing a scene.

**Thresholds were chosen after measuring the material, not before.** Every number below sits
against an observed distribution across the whole corpus; picking them first would have
produced categories that were either empty or universal. Two earlier attempts failed outright
and are worth recording: a "near maximum code value" test for clipped highlights matched
*nothing*, because S-Log3 never reaches full range; and a naive ``(max-min)/max`` saturation
exceeded 1 on saturated sources, because S-Gamut3.Cine to Rec.709 legitimately produces
negative channels.

Membership is not exclusive. A dark clip with saturated practicals belongs in both, and that
is the point — the interesting failures live where conditions overlap.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Category:
    """One validation condition."""

    id: str
    title: str
    provokes: str
    """The failure mode this category exists to expose."""

    rule: str
    """The membership rule, in words, matching the predicate."""

    predicate: Callable[[Mapping[str, Any]], bool]
    """Applied to a clip's measured statistics."""

    human_labelled: bool = False
    """True when membership cannot be decided by measurement alone."""


def _luma(measured: Mapping[str, Any], percentile: str) -> float:
    return float(measured["luma_p"][percentile])


CATEGORIES: tuple[Category, ...] = (
    Category(
        id="normal_daylight_interior",
        title="Normal daylight and interiors",
        provokes="baseline drift — a look that misbehaves on ordinary material is disqualified",
        rule="median linear luma between 0.02 and 1.0, little floor content, no blown highlights",
        predicate=lambda m: (
            0.02 <= _luma(m, "50") <= 1.0 and m["near_floor"] < 0.15 and _luma(m, "95") < 2.0
        ),
    ),
    Category(
        id="deep_underexposure",
        title="Deep underexposure",
        provokes="black crush, hue shifts in the toe, noise lifted into visible colour",
        rule="over 35% of pixels near the code floor, or median linear luma below 0.0015",
        predicate=lambda m: m["near_floor"] > 0.35 or _luma(m, "50") < 0.0015,
    ),
    Category(
        id="overexposure_clipped",
        title="Overexposure and clipped highlights",
        provokes="hard clipping, hue twist above white, loss of highlight separation",
        rule="95th-percentile linear luma above 1.5, or top code above 0.78",
        predicate=lambda m: _luma(m, "95") > 1.5 or m["code_max"] > 0.78,
    ),
    Category(
        id="saturated_practical",
        title="Saturated practical lights and objects",
        provokes="gamut collapse, channel clipping, hue rotation on out-of-gamut sources",
        rule="over 2% of pixels outside Rec.709, or broad high-saturation coverage at high luma",
        predicate=lambda m: (
            m["out_of_gamut"] > 0.02 or (m["sat_cov_040"] > 0.75 and m["chroma_hi_luma"] > 0.70)
        ),
    ),
    Category(
        id="low_saturation",
        title="Low-saturation scenes",
        provokes="false colour and tinting where there should be none; neutral-axis errors",
        rule="median saturation below 0.25 and under 35% of pixels above 0.40 saturation",
        predicate=lambda m: m["sat_p50"] < 0.25 and m["sat_cov_040"] < 0.35,
    ),
    Category(
        id="difficult_shadows",
        title="Difficult shadows",
        provokes="banding and gradient breakup where deep shadow meets a bright source",
        rule="over 13 stops between the 5th and 99th luma percentile, with real floor content",
        predicate=lambda m: m["dynamic_range_stops"] > 13.0 and m["near_floor"] > 0.15,
    ),
    Category(
        id="motion_or_noise",
        title="Motion and camera noise",
        provokes="temporal instability — a look that flickers or amplifies grain frame to frame",
        rule="frame-to-frame luma change above 0.9 relative, or dark-region noise above 0.12",
        predicate=lambda m: m["motion"] > 0.9 or m["noise"] > 0.12,
    ),
    # ---- skin: presence is human-labelled, lighting is measured -----------------
    Category(
        id="skin_neutral",
        title="Skin under neutral light",
        provokes="skin-tone accuracy on the reference case, with no colour cast to hide behind",
        rule="a labelled face, with a near-neutral frame cast",
        predicate=lambda m: _cast(m) == "neutral",
        human_labelled=True,
    ),
    Category(
        id="skin_warm",
        title="Skin under warm light",
        provokes="warm-cast skin drifting orange or losing separation from the background",
        rule="a labelled face, with a warm frame cast",
        predicate=lambda m: _cast(m) == "warm",
        human_labelled=True,
    ),
    Category(
        id="skin_green",
        title="Skin under green light",
        provokes="green-cast skin turning sallow; the hardest cast for a colour model",
        rule="a labelled face, with a green frame cast",
        predicate=lambda m: _cast(m) == "green",
        human_labelled=True,
    ),
    Category(
        id="skin_mixed",
        title="Skin under mixed lighting",
        provokes="competing sources pulling skin in two directions at once",
        rule="a labelled face, with two or more disagreeing casts in frame",
        predicate=lambda m: _cast(m) == "mixed",
        human_labelled=True,
    ),
)

BY_ID: Mapping[str, Category] = {category.id: category for category in CATEGORIES}

#: Clips containing a visible face, established by visual review of a full contact sheet.
#: Face *presence* is not measurable here yet — detection arrives with ``capabilities/detect``
#: — so this list is explicitly a human label and is marked as such in the catalogue.
FACE_CLIPS: frozenset[str] = frozenset(
    {
        "C0011",
        "C0040",
        "C0054",
        "C0055",
        "C0071",
        "C0072",
        "C0087",
        "C0113",
        "C0114",
        "C0115",
        "C0122",
    }
)

# Cast thresholds are corpus-relative, because this material has no white balance applied and
# therefore no absolutely neutral frame: measured warm/cool runs 0.22 to 1.15 across every face
# clip, so an absolute "near zero is neutral" test would classify all of them as warm. These
# cut points separate the observed population instead.
WARM_COOL_WARM = 0.45
GREEN_THRESHOLD = 0.05
MIXED_DISAGREEMENT = 0.90


def _cast(measured: Mapping[str, Any]) -> str:
    """The dominant colour cast of a frame, from its mean opponent axes.

    Sign convention: ``cast_warm_cool`` is ``(R-B)/Y`` so positive is warm;
    ``cast_green_magenta`` is ``(G - (R+B)/2)/Y`` so **positive is green** and negative is
    magenta. Getting that backwards labelled every magenta frame as green.
    """
    warm_cool = float(measured.get("cast_warm_cool", 0.0))
    green_magenta = float(measured.get("cast_green_magenta", 0.0))
    disagreement = float(measured.get("cast_disagreement", 0.0))
    if disagreement > MIXED_DISAGREEMENT:
        return "mixed"
    if green_magenta > GREEN_THRESHOLD:
        return "green"
    if warm_cool > WARM_COOL_WARM:
        return "warm"
    return "neutral"


def classify(measured: Mapping[str, Any], *, clip_id: str) -> list[str]:
    """Every category a clip belongs to, measured plus human-labelled."""
    assigned: list[str] = []
    for category in CATEGORIES:
        if category.human_labelled and clip_id not in FACE_CLIPS:
            continue
        try:
            if category.predicate(measured):
                assigned.append(category.id)
        except (KeyError, TypeError, ValueError):
            continue
    return assigned


def describe() -> list[dict[str, Any]]:
    """Serialisable taxonomy, for embedding in the catalogue manifest."""
    return [
        {
            "id": category.id,
            "title": category.title,
            "provokes": category.provokes,
            "rule": category.rule,
            "human_labelled": category.human_labelled,
        }
        for category in CATEGORIES
    ]


__all__ = ["BY_ID", "CATEGORIES", "FACE_CLIPS", "Category", "classify", "describe"]
