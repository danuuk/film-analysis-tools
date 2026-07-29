"""Content identity, artifact records and provenance — as utility, not as proof.

Content hashing here serves three purposes, none of which is verification ceremony
(``MIGRATION_PLAN.md`` sections 5.2 and 7):

1. **cache key** — sample once, reuse across hypotheses;
2. **change detection** — did the corpus move between these two runs?
3. **regeneration identity** — prove a re-extracted picture is the same picture, which is
   what makes deleting derived data safe.

The legacy contract specified roughly 40 wire schemas and never finished the packages meant
to consume them. About six survive here. Verification certificates, publication chains,
nonces and migration equivalence proofs are deliberately absent; they return only for a
named question, at ``Tier.FROZEN``.
"""

from __future__ import annotations

__all__: list[str] = []
