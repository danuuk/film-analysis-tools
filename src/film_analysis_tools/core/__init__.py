"""Foundation: errors, typed IO, workspace resolution, shared protocols, the tier ladder.

The bottom layer. Depends on nothing else in this package.

Replaces the legacy ``mediachar.core.contract`` — in-degree 153, whose ``fail()`` raised
``SystemExit`` from library code — and the hardcoded ``findings/`` paths in 90 modules and
``parents[N]`` root-walking in 39.

**Parallel and native-thread control is deliberately absent.** The legacy equivalent
(``mediachar.core.native_threads``, in-degree 44) capped BLAS threads by setting environment
variables at *module import time*, which only works if it runs before NumPy is imported — a
global side effect that is part of why those modules could not be used as libraries. There is
nothing to parallelise here yet, so building a worker pool now would be speculation. When a
parallel workload arrives, the environment capping belongs in ``cli`` as a pre-import concern
and the worker-plan resolution belongs here as a pure function. See ``MIGRATION_PLAN.md``
section 2.2.
"""

from __future__ import annotations

from film_analysis_tools.core import io
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
    "io",
]
