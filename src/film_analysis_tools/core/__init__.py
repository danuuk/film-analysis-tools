"""Foundation: errors, workspace resolution, shared protocols, the tier ladder.

The bottom layer. Depends on nothing else in this package.

Replaces the legacy ``mediachar.core.contract`` (in-degree 153, 107 lines of JSON/CSV IO
whose ``fail()`` raised ``SystemExit`` from library code) and the hardcoded ``findings/``
output paths in 90 modules.

Typed IO and parallel/thread control are hardened in P3, shaped by what the fast path
actually needed rather than guessed upfront.
"""

from __future__ import annotations

from film_analysis_tools.core.errors import (
    ControlError,
    DataError,
    FilmAnalysisError,
    SelectionError,
    WorkspaceError,
)
from film_analysis_tools.core.protocols import RGB, Transform
from film_analysis_tools.core.tiers import REQUIRED_CONTROLS, Tier
from film_analysis_tools.core.workspace import Workspace

__all__ = [
    "REQUIRED_CONTROLS",
    "RGB",
    "ControlError",
    "DataError",
    "FilmAnalysisError",
    "SelectionError",
    "Tier",
    "Transform",
    "Workspace",
    "WorkspaceError",
]
