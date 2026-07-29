"""Comparing two transforms over a cohort.

The output is a distribution summary, never a verdict: effect, spread, sample count, and the
null-control result, with the tier the comparison is entitled to claim at. Robust statistics
throughout — median and MAD rather than mean and standard deviation — matching the quantile
and robust-scale style already dominant in the legacy analysis (57 and 37 modules).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from film_analysis_tools.capabilities.colour import metrics as metrics_module
from film_analysis_tools.capabilities.sample.table import SampleTable
from film_analysis_tools.capabilities.statistics.controls import (
    DEFAULT_RESAMPLES,
    NullResult,
    shuffled_labels,
)
from film_analysis_tools.core.errors import ControlError, DataError
from film_analysis_tools.core.protocols import Transform
from film_analysis_tools.core.tiers import REQUIRED_CONTROLS, Tier

MAD_TO_SIGMA = 1.4826


#: A signed effect below this share of the magnitude means per-sample changes cancel rather
#: than agreeing on a direction. Stated as a convention, not derived from theory.
DIRECTIONAL_SHARE = 0.5

EPS = 1.0e-12


@dataclass(frozen=True)
class Comparison:
    """One cohort, one metric, two transforms."""

    cohort: str
    metric: str
    unit: str
    effect: float
    """Median signed change — the *directional* tendency."""

    magnitude: float
    """Median absolute change — how much samples move at all, direction ignored."""

    spread: float
    count: int
    null: NullResult
    tier: Tier

    @property
    def exceeds_null(self) -> bool:
        """Whether the signed effect stands clear of its own null distribution."""
        return abs(self.effect) > max(self.null.spread, EPS)

    @property
    def is_directional(self) -> bool:
        """Whether per-sample changes agree on a direction, or cancel out.

        A per-channel gain over a full hue circle moves every sample yet nets to zero,
        because samples on opposite sides of the circle move opposite ways. Reporting that
        as "no effect" would be wrong; it is "no *net* effect on this cohort", and it is
        usually a sign the cohort is too broad for the question.
        """
        return abs(self.effect) >= DIRECTIONAL_SHARE * self.magnitude

    @property
    def verdict(self) -> str:
        if self.magnitude <= EPS:
            return "no change"
        if not self.is_directional:
            return "moves, no net direction"
        return "clear of null" if self.exceeds_null else "within null"

    def as_record(self) -> dict[str, Any]:
        """A flat, JSON- and CSV-friendly record. Reports the components, never a lone verdict."""
        return {
            "cohort": self.cohort,
            "metric": self.metric,
            "unit": self.unit,
            "effect": self.effect,
            "magnitude": self.magnitude,
            "spread": self.spread,
            "count": self.count,
            "null_effect": self.null.effect,
            "null_spread": self.null.spread,
            "null_p_value": self.null.p_value,
            "null_resamples": self.null.resamples,
            "null_is_clean": self.null.is_clean,
            "is_directional": self.is_directional,
            "exceeds_null": self.exceeds_null,
            "verdict": self.verdict,
            "tier": self.tier.value,
        }

    def summary(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        return (
            f"{self.cohort:<28} {self.metric:<17} "
            f"{self.effect:+.4f}{unit}  "
            f"(|move| {self.magnitude:.4f}, spread {self.spread:.4f}, n={self.count:,})  "
            f"null {self.null.effect:+.4f} p={self.null.p_value:.3f}  "
            f"{self.verdict}  [{self.tier.value}]"
        )


def per_sample_metric(
    cohort: SampleTable,
    *,
    baseline: Transform,
    candidate: Transform,
    metric: str | metrics_module.Metric = "hue_drift",
) -> np.ndarray:
    """The finite per-row metric values behind a comparison.

    Exposed because the distribution is the thing worth looking at: a summary can only say
    the median cancelled, while the distribution shows *why* — two lobes of opposite sign,
    a long tail, or a clean directional shift.
    """
    metric_fn = metrics_module.named(metric) if isinstance(metric, str) else metric
    rgb = cohort.rgb
    values = np.asarray(metric_fn(baseline(rgb), candidate(rgb)), dtype=np.float64)
    return values[np.isfinite(values)]


def compare(
    cohort: SampleTable,
    *,
    baseline: Transform,
    candidate: Transform,
    metric: str | metrics_module.Metric = "hue_drift",
    tier: Tier = Tier.COMPARISON,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    label: str | None = None,
) -> Comparison:
    """Measure how ``candidate`` differs from ``baseline`` over ``cohort``.

    The null control always runs. It is not an option because a comparison that cannot show
    what "no effect" looks like is not evidence.
    """
    if len(cohort) == 0:
        raise DataError(f"cohort {cohort.label!r} is empty; nothing to compare")

    metric_name = metric if isinstance(metric, str) else getattr(metric, "__name__", "metric")
    metric_fn = metrics_module.named(metric) if isinstance(metric, str) else metric

    required = REQUIRED_CONTROLS[tier]
    unsupported = required - {"null"}
    if unsupported:
        raise ControlError(
            f"tier {tier.value!r} additionally requires {sorted(unsupported)}, "
            "which arrive with the controls work in P4"
        )

    finite = per_sample_metric(cohort, baseline=baseline, candidate=candidate, metric=metric_fn)
    if finite.size == 0:
        raise DataError(f"metric {metric_name!r} produced no finite values on {cohort.label!r}")

    effect = float(np.median(finite))
    magnitude = float(np.median(np.abs(finite)))
    spread = float(MAD_TO_SIGMA * np.median(np.abs(finite - effect)))
    null = shuffled_labels(finite, effect, resamples=resamples, seed=seed)

    return Comparison(
        cohort=label if label is not None else cohort.label,
        metric=metric_name,
        unit=metrics_module.UNITS.get(metric_name, ""),
        effect=effect,
        magnitude=magnitude,
        spread=spread,
        count=int(finite.size),
        null=null,
        tier=tier,
    )


def compare_cohorts(
    cohorts: dict[str, SampleTable],
    *,
    baseline: Transform,
    candidate: Transform,
    metric: str | metrics_module.Metric = "hue_drift",
    tier: Tier = Tier.COMPARISON,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> list[Comparison]:
    """The same comparison across several cohorts — the shape most questions take."""
    return [
        compare(
            table,
            baseline=baseline,
            candidate=candidate,
            metric=metric,
            tier=tier,
            resamples=resamples,
            seed=seed,
            label=name,
        )
        for name, table in cohorts.items()
    ]


__all__ = ["MAD_TO_SIGMA", "Comparison", "compare", "compare_cohorts"]
