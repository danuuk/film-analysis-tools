"""Statistics primitives: patch grids, accumulators, PSD, noise, bucketing.

Engine-free by construction. Nothing here needs a forward model or a reference.
"""

from __future__ import annotations

from film_analysis_tools.capabilities.measure import (
    chroma,
    evidence,
    residual,
    synthetic,
    windows,
)
from film_analysis_tools.capabilities.measure.residual import ResidualEstimate, extract
from film_analysis_tools.capabilities.measure.synthetic import SyntheticSpec
from film_analysis_tools.capabilities.measure.windows import (
    SelectionReport,
    Window,
    WindowGate,
    select_windows,
)

__all__ = [
    "ResidualEstimate",
    "SelectionReport",
    "SyntheticSpec",
    "Window",
    "WindowGate",
    "chroma",
    "evidence",
    "extract",
    "residual",
    "select_windows",
    "synthetic",
    "windows",
]
