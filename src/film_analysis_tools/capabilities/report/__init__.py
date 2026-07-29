"""Summaries that lead with effect size and spread.

A pass/fail verdict is the shape that hides a marginal mechanism, so reports state
the effect, its spread, the sample count and the null-control result.
"""

from __future__ import annotations

from film_analysis_tools.capabilities.report.summary import format_comparisons, format_pack

__all__ = ["format_comparisons", "format_pack"]
