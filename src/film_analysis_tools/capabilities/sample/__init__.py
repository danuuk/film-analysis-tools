"""The tidy sample table, cohorts and selectors.

A sample pack is roughly 260k rows of ~17 named columns, about 12 MB. Sample once,
select many times: a cohort is a named query, not a bespoke pipeline. Carries
per-row provenance (source frame, source pixel) so a sample never loses its basis.
"""

from __future__ import annotations

__all__: list[str] = []
