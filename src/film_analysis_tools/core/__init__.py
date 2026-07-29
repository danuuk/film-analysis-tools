"""Foundation: errors, typed IO, workspace resolution, parallel control, shared protocols.

The bottom layer. Depends on nothing else in this package.

Replaces the legacy ``mediachar.core.contract`` (in-degree 153, 107 lines of JSON/CSV IO
whose ``fail()`` raised ``SystemExit`` from library code) and the hardcoded ``findings/``
output paths in 90 modules.

Planned contents (P3): ``errors``, ``io``, ``workspace``, ``parallel``. The ``ForwardModel``
protocol also lands here rather than in ``forward/`` so that ``capabilities.fit`` can depend
on the abstraction without depending on any adapter — see ``ARCHITECTURE.md``.
"""

from __future__ import annotations

from film_analysis_tools.core.tiers import REQUIRED_CONTROLS, Tier

__all__ = ["REQUIRED_CONTROLS", "Tier"]
