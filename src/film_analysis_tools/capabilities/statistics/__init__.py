"""Statistical tests and the controls that keep them honest.

Controls are first-class, not optional extras: null/permutation, trivial baseline,
holdout, perturbation. A comparison without a null control cannot claim a tendency
(MIGRATION_PLAN.md section 2.3).
"""

from __future__ import annotations

from film_analysis_tools.capabilities.statistics.compare import Comparison, compare, compare_cohorts
from film_analysis_tools.capabilities.statistics.controls import NullResult, shuffled_labels

__all__ = ["Comparison", "NullResult", "compare", "compare_cohorts", "shuffled_labels"]
