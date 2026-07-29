"""Rule 7: a study may not claim above its tier (``MIGRATION_PLAN.md`` section 10).

This is the one rule that replaces the legacy verification-certificate machinery. Elaborate,
formal, hard-to-reproduce procedure invites unearned confidence in its result; the legacy
apparatus never revealed that the approach under test worked only under limited conditions.
Honesty by label costs one field and is read every time.

The study-discovery half is inert until ``studies/`` gains content in P11, and deliberately
so: it fails closed the moment a study appears without a declared tier, rather than being
retrofitted later. The vocabulary half has teeth now.
"""

from __future__ import annotations

import pkgutil

import film_analysis_tools.studies as studies_package
from film_analysis_tools.core.tiers import REQUIRED_CONTROLS, Tier


def _declared_studies() -> list[object]:
    found: list[object] = []
    for module_info in pkgutil.walk_packages(
        studies_package.__path__, prefix=f"{studies_package.__name__}."
    ):
        module = __import__(module_info.name, fromlist=["_"])
        found.extend(
            value
            for name, value in vars(module).items()
            if not name.startswith("_") and hasattr(value, "tier") and hasattr(value, "question")
        )
    return found


def test_tier_ladder_is_totally_ordered() -> None:
    tiers = list(Tier)
    assert tiers == sorted(tiers, key=lambda tier: tier.rank)
    assert Tier.PROBE < Tier.COMPARISON < Tier.STUDY < Tier.FROZEN
    assert len({tier.rank for tier in tiers}) == len(tiers)


def test_required_controls_are_declared_for_every_tier_and_never_weaken() -> None:
    assert set(REQUIRED_CONTROLS) == set(Tier)
    for lower, higher in zip(list(Tier), list(Tier)[1:], strict=False):
        assert REQUIRED_CONTROLS[lower] <= REQUIRED_CONTROLS[higher], (
            f"{higher.value} requires fewer controls than {lower.value}"
        )


def test_comparison_tier_requires_a_null_control() -> None:
    """The cheapest check that catches a broken test, and the legacy system's weakest link
    at 3 modules. It is required from the default tier upward."""
    assert "null" in REQUIRED_CONTROLS[Tier.COMPARISON]
    assert not REQUIRED_CONTROLS[Tier.PROBE]


def test_every_declared_study_states_a_tier_and_carries_its_required_controls() -> None:
    for study in _declared_studies():
        tier = study.tier
        assert isinstance(tier, Tier), f"{study!r} must declare a Tier"
        declared = {str(control) for control in getattr(study, "controls", ())}
        missing = REQUIRED_CONTROLS[tier] - declared
        assert not missing, f"{study!r} claims {tier.value} without controls: {sorted(missing)}"
        if tier > Tier.PROBE:
            for field in ("question", "rationale", "assumptions", "falsified_by"):
                assert getattr(study, field, None), f"{study!r} at {tier.value} needs {field}"
