"""Controls: the cheap independent checks that catch a broken test.

The highest-value one is the **null control** — run the test where the answer must be "no
effect". If it fires, the test is broken, not the mechanism. It costs almost nothing and it
is more informative than any certificate chain. In the legacy system this appeared in three
modules out of 221, which is why it is required here from the default tier upward.

For a paired before/after design the null is label exchange: randomly flip which side of each
pair counts as "candidate". A real effect survives nothing of the kind; a bug in the metric
or the pairing usually does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 200

#: Peak working-set budget for the resampling matrix, in bytes. Resamples are drawn in
#: batches sized to fit, so memory is bounded by this rather than by ``resamples x n``.
#: Drawing in batches consumes the generator in exactly the same order as one large draw,
#: so results are bit-identical to the unbatched version.
DEFAULT_MEMORY_BUDGET_BYTES = 64_000_000


@dataclass(frozen=True)
class NullResult:
    """What the same measurement produces when the effect has been destroyed by design."""

    effect: float
    """Typical effect size under label exchange. Should sit near zero."""

    spread: float
    """Spread of the null distribution — the scale an observed effect must beat."""

    p_value: float
    """Fraction of resamples whose effect was at least as extreme as the observed one."""

    resamples: int

    @property
    def is_clean(self) -> bool:
        """True when the null landed near zero, as a well-formed null control should."""
        return abs(self.effect) <= max(self.spread, 1.0e-12)


def shuffled_labels(
    per_sample: np.ndarray,
    observed: float,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> NullResult:
    """Permutation null for a paired difference, by random sign flip.

    ``per_sample`` holds one signed value per row (candidate minus baseline). Flipping signs
    at random is exactly the exchange of the two labels, so the resulting distribution is
    what this metric produces when the two renderings are interchangeable.

    Resamples are drawn in batches sized to ``memory_budget_bytes``. Drawn in one block, a
    260k-row cohort at 200 resamples peaks above a gigabyte and grows linearly with the
    corpus; batching bounds it at roughly the budget while consuming the random stream in the
    same order, so the result is unchanged.
    """
    values = np.asarray(per_sample, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return NullResult(effect=0.0, spread=0.0, p_value=1.0, resamples=0)

    generator = np.random.default_rng(seed)
    pool = np.asarray([-1.0, 1.0])
    row_bytes = max(1, values.size * values.itemsize)
    batch = max(1, min(resamples, memory_budget_bytes // row_bytes))

    effects = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, batch):
        size = min(batch, resamples - start)
        signs = generator.choice(pool, size=(size, values.size))
        np.multiply(signs, values, out=signs)
        effects[start : start + size] = np.median(signs, axis=1)

    at_least_as_extreme = int(np.count_nonzero(np.abs(effects) >= abs(observed)))
    return NullResult(
        effect=float(np.median(effects)),
        spread=float(np.std(effects)),
        # +1 in both terms so a null that never reaches the observed effect reports a
        # bounded p rather than an unearned exact zero.
        p_value=float((at_least_as_extreme + 1) / (resamples + 1)),
        resamples=resamples,
    )


__all__ = ["DEFAULT_RESAMPLES", "NullResult", "shuffled_labels"]
