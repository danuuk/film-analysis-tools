"""The escalation ladder: how much a result is entitled to claim.

Rigour is opt-in and escalating (``MIGRATION_PLAN.md`` section 2.5). A study declares the
tier it ran at, and may not claim beyond it — that single rule replaces the verification
certificate machinery the legacy system accumulated.
"""

from __future__ import annotations

from enum import Enum


class Tier(Enum):
    """Ordered rigour levels. Compare with ``<``/``>``; order is definition order."""

    PROBE = "probe"
    """Minutes, disposable, no artifacts retained. May claim only that a mechanism
    responds or does not respond."""

    COMPARISON = "comparison"
    """The default. A null control is required. May claim a tendency, on this cohort."""

    STUDY = "study"
    """Adds holdout and perturbation, and retains artifacts. May claim the effect holds
    beyond the fitted data."""

    FROZEN = "frozen"
    """Adds full provenance and reproducibility from a pin. May act as an authority that
    other work builds on. Never reached by default."""

    @property
    def rank(self) -> int:
        return _ORDER.index(self)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return self.rank >= other.rank


_ORDER: tuple[Tier, ...] = (Tier.PROBE, Tier.COMPARISON, Tier.STUDY, Tier.FROZEN)

#: Controls a tier requires before a result may claim at that level. Enforced from P4.
REQUIRED_CONTROLS: dict[Tier, frozenset[str]] = {
    Tier.PROBE: frozenset(),
    Tier.COMPARISON: frozenset({"null"}),
    Tier.STUDY: frozenset({"null", "holdout", "perturbation"}),
    Tier.FROZEN: frozenset({"null", "holdout", "perturbation"}),
}

__all__ = ["REQUIRED_CONTROLS", "Tier"]
