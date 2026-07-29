"""Statistics primitives: patch grids, accumulators, PSD, noise, bucketing.

Engine-free by construction. Nothing here needs a forward model or a reference.
"""

from __future__ import annotations

from film_analysis_tools.capabilities.measure import residual, synthetic
from film_analysis_tools.capabilities.measure.residual import ResidualEstimate, extract
from film_analysis_tools.capabilities.measure.synthetic import SyntheticSpec

__all__ = ["ResidualEstimate", "SyntheticSpec", "extract", "residual", "synthetic"]
