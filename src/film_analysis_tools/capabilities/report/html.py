"""Self-contained HTML reports.

One file, no assets, no network. It opens in a browser, survives being emailed, and carries
enough provenance to answer "what was this run, on what, with which controls" months later —
which a terminal table scrolled off the screen cannot.

The page states its tier and its controls prominently. A report that looks authoritative while
resting on a single marginal metric is the failure this system is being rebuilt against, so the
limits travel with the figures rather than in a footnote.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from film_analysis_tools.capabilities.report import charts, svg
from film_analysis_tools.capabilities.sample.table import SampleTable
from film_analysis_tools.capabilities.statistics.compare import Comparison
from film_analysis_tools.core.protocols import Transform

STYLE = """
:root {
  --bg: #ffffff; --ink: #16181d; --muted: #6b7280; --grid: #d6dae1;
  --accent: #2563eb; --baseline: #6b7280; --null-band: #cbd5e1;
  --card: #f7f8fa; --border: #e5e7eb; --warn: #b45309;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --ink: #e8eaed; --muted: #9aa3af; --grid: #333941;
    --accent: #7aa2f7; --baseline: #9aa3af; --null-band: #2b3138;
    --card: #1b1e24; --border: #2a2e35; --warn: #e0a458;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 28px; background: var(--bg); color: var(--ink);
  font: 14px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1120px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 32px 0 10px; padding-bottom: 6px;
  border-bottom: 1px solid var(--border); }
.sub { color: var(--muted); margin: 0 0 18px; }
.meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 10px; margin: 0 0 8px; }
.meta div { background: var(--card); border: 1px solid var(--border);
  border-radius: 7px; padding: 9px 11px; }
.meta dt { color: var(--muted); font-size: 11px; text-transform: uppercase;
  letter-spacing: .04em; margin: 0; }
.meta dd { margin: 3px 0 0; font-weight: 600; word-break: break-word; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 16px; }
.fig { margin: 0; background: var(--card); border: 1px solid var(--border);
  border-radius: 9px; padding: 12px; overflow-x: auto; }
.fig figcaption { display: flex; flex-direction: column; gap: 2px; margin-bottom: 6px; }
.fig figcaption span { color: var(--muted); font-size: 12px; }
.chart { width: 100%; height: auto; display: block; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums;
  font-size: 13px; }
th, td { text-align: right; padding: 6px 9px; border-bottom: 1px solid var(--border);
  white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase;
  letter-spacing: .04em; }
.wrap { overflow-x: auto; }
.note { border-left: 3px solid var(--warn); background: var(--card); padding: 9px 13px;
  border-radius: 0 7px 7px 0; margin: 14px 0; color: var(--ink); }
.tier { display: inline-block; background: var(--accent); color: #fff; border-radius: 999px;
  padding: 1px 9px; font-size: 11px; font-weight: 700; letter-spacing: .03em; }
.empty { color: var(--muted); font-style: italic; margin: 12px 0; }
footer { color: var(--muted); font-size: 12px; margin-top: 34px;
  border-top: 1px solid var(--border); padding-top: 12px; }
"""


@dataclass(frozen=True)
class ReportContext:
    """Everything a reader needs to know what produced the figures."""

    title: str
    pack: str
    baseline: str
    candidate: str
    metric: str
    resamples: int
    seed: int
    roots: Mapping[str, str]


def _meta(context: ReportContext, total: int) -> str:
    fields = [
        ("dataset", context.pack),
        ("baseline", context.baseline),
        ("candidate", context.candidate),
        ("metric", context.metric),
        ("samples", f"{total:,}"),
        ("null resamples", f"{context.resamples} (seed {context.seed})"),
        ("read root", context.roots.get("read_root", "")),
        ("write root", context.roots.get("write_root", "") or "—"),
    ]
    cells = "".join(
        f"<div><dt>{svg.escape(name)}</dt><dd>{svg.escape(value)}</dd></div>"
        for name, value in fields
        if value
    )
    return f'<dl class="meta">{cells}</dl>'


def _table(results: Sequence[Comparison]) -> str:
    head = (
        "<tr><th>cohort</th><th>effect</th><th>|move|</th><th>spread</th><th>n</th>"
        "<th>null</th><th>p</th><th>verdict</th></tr>"
    )
    rows = "".join(
        f"<tr><td>{svg.escape(result.cohort)}</td>"
        f"<td>{result.effect:+.4f}</td><td>{result.magnitude:.4f}</td>"
        f"<td>{result.spread:.4f}</td><td>{result.count:,}</td>"
        f"<td>{result.null.effect:+.4f}</td><td>{result.null.p_value:.3f}</td>"
        f"<td>{svg.escape(result.verdict)}</td></tr>"
        for result in results
    )
    return f'<div class="wrap"><table>{head}{rows}</table></div>'


def _notes(results: Sequence[Comparison]) -> str:
    parts: list[str] = []
    cancelling = [r.cohort for r in results if not r.is_directional]
    if cancelling:
        parts.append(
            "<div class='note'>Samples move but cancel out in "
            f"<b>{svg.escape(', '.join(cancelling))}</b>. The effect has no net direction on "
            "these cohorts — usually a sign the cohort is too broad for the question. The hue "
            "response curve shows where the sign changes.</div>"
        )
    dirty = [r.cohort for r in results if not r.null.is_clean]
    if dirty:
        parts.append(
            "<div class='note'>The null control did not land near zero for "
            f"<b>{svg.escape(', '.join(dirty))}</b>. Treat those effects as unproven and check "
            "the metric and the pairing before reading anything into them.</div>"
        )
    return "".join(parts)


def comparison_report(
    context: ReportContext,
    results: Sequence[Comparison],
    cohorts: Mapping[str, SampleTable],
    per_sample: Mapping[str, np.ndarray],
    *,
    baseline: Transform,
    candidate: Transform,
) -> str:
    """Assemble the full page."""
    tiers = sorted({result.tier.value for result in results})
    total = sum(result.count for result in results)

    distributions = "".join(
        charts.metric_distribution(per_sample.get(result.cohort, np.asarray([])), result)
        for result in results
    )
    coverage = "".join(charts.cohort_coverage(table, title=name) for name, table in cohorts.items())
    samples = "".join(
        charts.sample_swatches(table, baseline=baseline, candidate=candidate, title=name)
        for name, table in cohorts.items()
    )

    body = f"""<main>
<h1>{svg.escape(context.title)}</h1>
<p class="sub"><span class="tier">{svg.escape(", ".join(tiers))}</span>
&nbsp;a tendency on these cohorts, not a fact beyond them. Every comparison below ran a
permutation null control; effect sizes are medians and spreads are robust (MAD-scaled).</p>
{_meta(context, total)}
{_notes(results)}
<h2>Results</h2>
{_table(results)}
<h2>What the transform does</h2>
<p class="sub">Independent of any corpus — swept inputs through both transforms.</p>
<div class="grid">{charts.tone_response(baseline, candidate)}
{charts.hue_response(baseline, candidate)}</div>
<h2>Metric distributions</h2>
<p class="sub">Shaded band is the null spread; the solid line is the observed median.</p>
<div class="grid">{distributions}</div>
<h2>Cohort coverage</h2>
<p class="sub">Where each cohort actually sits, so a selector can be checked rather than
trusted because of its name.</p>
<div class="grid">{coverage}</div>
<h2>Samples</h2>
<p class="sub">Representative colours before and after, encoded to sRGB for display only.</p>
<div class="grid">{samples}</div>
<footer>Generated {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")} by film-analysis-tools.
Colour-defined cohorts are named <code>_like</code> because they describe a region of colour
space, not a detected subject.</footer>
</main>"""

    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{svg.escape(context.title)}</title><style>{STYLE}</style></head>"
        f"<body>{body}</body></html>\n"
    )


__all__ = ["ReportContext", "comparison_report"]
