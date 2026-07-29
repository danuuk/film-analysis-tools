"""Adapters onto emulation models — the single seam against engine and plugin drift.

This is the **only** layer permitted to import an emulation model. That is a maintenance
boundary, not a wall: this package is *expected* to depend on the engine and the plugin
runtime. Confining it to one place means one seam to maintain rather than twenty, which is
what the legacy code had.

The ``ForwardModel`` protocol itself lives in ``core`` so that ``capabilities.fit`` can
depend on the abstraction without depending on an adapter.

Planned adapters (P8): the engine, the plugin runtime, a pinned legacy oracle for
reproducing historical results, and a synthetic analytic model whose answers are known in
closed form.
"""

from __future__ import annotations

__all__: list[str] = []
