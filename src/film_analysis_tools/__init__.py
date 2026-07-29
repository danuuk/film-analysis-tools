"""Film Analysis Tools — measurement, fitting and validation for film emulation work.

Two contours of validation (``MIGRATION_PLAN.md`` section 1.1):

* **Contour A — fidelity.** Does the emulation model match the reference? This repo's job.
  Needs corpus, catalogue, sampling and statistics.
* **Contour B — well-formedness.** Is the transform itself sound? That lives in the engine,
  where it can gate every mechanism change on synthetic input with no corpus.

Layering is strictly downward; see ``ARCHITECTURE.md``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
