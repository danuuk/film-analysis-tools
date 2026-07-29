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
) -> NullResult:
    """Permutation null for a paired difference, by random sign flip.

    ``per_sample`` holds one signed value per row (candidate minus baseline). Flipping signs
    at random is exactly the exchange of the two labels, so the resulting distribution is
    what this metric produces when the two renderings are interchangeable.
    """
    values = np.asarray(per_sample, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return NullResult(effect=0.0, spread=0.0, p_value=1.0, resamples=0)

    generator = np.random.default_rng(seed)
    signs = generator.choice(np.asarray([-1.0, 1.0]), size=(resamples, values.size))
    effects = np.median(signs * values, axis=1)

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
