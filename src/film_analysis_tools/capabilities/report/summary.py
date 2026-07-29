"""Rendering comparison results for a human.

Reports lead with effect size, spread and sample count, and always show the null control. A
bare verdict is the shape that hides a marginal mechanism, so there isn't one — the reader
sees the size of the move and the size of the noise it has to beat.
"""

from __future__ import annotations

from collections.abc import Sequence

from film_analysis_tools.capabilities.statistics.compare import Comparison

HEADER = (
    f"{'cohort':<28} {'metric':<17} {'effect':>12}  {'|move|':>10}  "
    f"{'spread':>10}  {'n':>9}  {'null':>10}  {'p':>6}  verdict"
)


def format_comparisons(results: Sequence[Comparison], *, title: str = "") -> str:
    """A fixed-width table of comparison results."""
    if not results:
        return "no cohorts produced any samples"

    lines: list[str] = []
    if title:
        lines.extend((title, "=" * len(title)))
    lines.append(HEADER)
    lines.append("-" * len(HEADER))
    for result in results:
        unit = f" {result.unit}" if result.unit else ""
        lines.append(
            f"{result.cohort:<28} {result.metric:<17} "
            f"{result.effect:>+11.4f}{unit:<1}  {result.magnitude:>10.4f}  "
            f"{result.spread:>10.4f}  {result.count:>9,}  "
            f"{result.null.effect:>+10.4f}  {result.null.p_value:>6.3f}  {result.verdict}"
        )

    tiers = {result.tier.value for result in results}
    lines.append("")
    lines.append(
        f"tier: {', '.join(sorted(tiers))} — a tendency on these cohorts, not a fact beyond them"
    )

    cancelling = [result.cohort for result in results if not result.is_directional]
    if cancelling:
        lines.append(
            f"NOTE: samples move but cancel out in {cancelling} — the effect has no net "
            "direction here. Usually the cohort is too broad for the question; split it."
        )
    dirty = [result.cohort for result in results if not result.null.is_clean]
    if dirty:
        lines.append(
            f"WARNING: null control did not land near zero for {dirty}; "
            "treat these effects as unproven and check the metric or the pairing"
        )
    return "\n".join(lines)


def format_pack(description: dict[str, object]) -> str:
    """A short description of a sample pack."""
    lines = [
        f"pack       {description['name']}",
        f"rows       {description['rows']:,}",
        f"scenes     {description['scenes']}",
    ]
    for key in ("pack_id", "role", "generated_at_utc"):
        value = description.get(key)
        if value:
            lines.append(f"{key:<10} {value}")
    columns = description.get("columns")
    if isinstance(columns, list):
        lines.append(f"columns    {', '.join(str(column) for column in columns)}")
    return "\n".join(lines)


__all__ = ["HEADER", "format_comparisons", "format_pack"]
