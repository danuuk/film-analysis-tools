"""The null control must stay bounded in memory — and identical in result.

Drawn as one block, the resampling matrix is ``resamples x n`` floats: over a gigabyte for a
260k-row cohort at 200 resamples, growing linearly with the corpus. The legacy system lost days
to exactly this class of allocation. Batching bounds it, and because a batched draw consumes
the generator in the same order, the numbers do not move.
"""

from __future__ import annotations

import tracemalloc

import numpy as np

from film_analysis_tools.capabilities.statistics.controls import (
    DEFAULT_MEMORY_BUDGET_BYTES,
    shuffled_labels,
)


def test_batching_does_not_change_the_result() -> None:
    """A budget large enough for one block must give bit-identical numbers to a small one."""
    values = np.random.default_rng(11).normal(size=20_000)
    unbatched = shuffled_labels(values, 0.01, resamples=64, seed=5, memory_budget_bytes=10**9)
    batched = shuffled_labels(values, 0.01, resamples=64, seed=5, memory_budget_bytes=200_000)
    assert batched.effect == unbatched.effect
    assert batched.spread == unbatched.spread
    assert batched.p_value == unbatched.p_value


def test_peak_memory_stays_near_the_budget_as_the_cohort_grows() -> None:
    budget = DEFAULT_MEMORY_BUDGET_BYTES
    peaks: list[int] = []
    for size in (25_000, 200_000):
        values = np.random.default_rng(3).normal(size=size)
        tracemalloc.start()
        shuffled_labels(values, 0.0, resamples=200, seed=1)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)

    # An eightfold cohort must not cost eightfold memory.
    assert peaks[1] < peaks[0] * 3
    # And the peak must stay within a small multiple of the declared budget.
    assert peaks[1] < budget * 4


def test_a_tiny_budget_still_produces_every_resample() -> None:
    values = np.random.default_rng(7).normal(size=5_000)
    result = shuffled_labels(values, 0.0, resamples=37, seed=2, memory_budget_bytes=1)
    assert result.resamples == 37
    assert result.p_value > 0.0


def test_an_empty_cohort_is_handled_rather_than_dividing_by_zero() -> None:
    result = shuffled_labels(np.asarray([np.nan, np.inf]), 0.0, resamples=10)
    assert result.resamples == 0
    assert result.p_value == 1.0
