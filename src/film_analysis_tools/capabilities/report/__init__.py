"""Summaries that lead with effect size and spread.

A pass/fail verdict is the shape that hides a marginal mechanism, so reports state
the effect, its spread, the sample count and the null-control result.
"""

from __future__ import annotations

from film_analysis_tools.capabilities.report import charts, html, svg
from film_analysis_tools.capabilities.report.html import ReportContext, comparison_report
from film_analysis_tools.capabilities.report.summary import format_comparisons, format_pack

__all__ = [
    "ReportContext",
    "charts",
    "comparison_report",
    "format_comparisons",
    "format_pack",
    "html",
    "svg",
]
