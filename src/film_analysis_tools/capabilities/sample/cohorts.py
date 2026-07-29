"""Named cohorts — reusable, comparable slices of a sample table.

A cohort is a named query so that two studies asking about "neutrals" mean the same thing,
and so a result can say which population it applies to without re-deriving it.

**These are colour-defined, not semantic.** ``skin_like`` selects the hue, saturation and luma
region that skin usually occupies; it does not know what skin is. True skin selection needs
detection and arrives with ``capabilities/detect`` in P7. The name says ``_like`` because
claiming otherwise is exactly the kind of unearned confidence this system is built to avoid.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from film_analysis_tools.capabilities.sample.table import SampleTable

CohortSelector = Callable[[SampleTable], SampleTable]


def neutral(table: SampleTable) -> SampleTable:
    """Near-achromatic samples — where a colour transform's neutral axis behaviour shows."""
    return table.where(relative_saturation__lt=0.05)


def skin_like(table: SampleTable) -> SampleTable:
    """The warm, moderately saturated, mid-luma region skin typically occupies."""
    return table.where(
        hue_sector__in=("red", "orange"),
        relative_saturation__between=(0.10, 0.60),
        luma_bt2020__between=(0.05, 0.70),
    )


def foliage_like(table: SampleTable) -> SampleTable:
    """Saturated greens."""
    return table.where(hue_sector="green", relative_saturation__gt=0.15)


def sky_like(table: SampleTable) -> SampleTable:
    """Saturated blues and cyans."""
    return table.where(hue_sector__in=("blue", "cyan"), relative_saturation__gt=0.15)


def shadows(table: SampleTable) -> SampleTable:
    return table.where(luma_bt2020__lt=0.08)


def highlights(table: SampleTable) -> SampleTable:
    """Bright samples — where collapse and clipping show up first."""
    return table.where(luma_bt2020__gt=0.70)


def saturated(table: SampleTable) -> SampleTable:
    return table.where(relative_saturation__gt=0.55)


BUILT_INS: Mapping[str, CohortSelector] = {
    "neutral": neutral,
    "skin_like": skin_like,
    "foliage_like": foliage_like,
    "sky_like": sky_like,
    "shadows": shadows,
    "highlights": highlights,
    "saturated": saturated,
}


def build(table: SampleTable, names: tuple[str, ...]) -> dict[str, SampleTable]:
    """Build several named cohorts from one table, skipping any that come out empty."""
    from film_analysis_tools.core.errors import SelectionError

    unknown = [name for name in names if name not in BUILT_INS]
    if unknown:
        raise SelectionError(f"unknown cohorts {unknown}; available: {sorted(BUILT_INS)}")
    built = {name: BUILT_INS[name](table) for name in names}
    return {name: cohort for name, cohort in built.items() if len(cohort) > 0}


__all__ = [
    "BUILT_INS",
    "CohortSelector",
    "build",
    "foliage_like",
    "highlights",
    "neutral",
    "saturated",
    "shadows",
    "skin_like",
    "sky_like",
]
