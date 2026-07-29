"""Colour features, source adapters, renderers, tile classification.

Feature extraction is the clean substrate the fast path is built on: hue, saturation,
luma, opponent axes, chroma headroom, and fixed-cell binning.
"""

from __future__ import annotations

from film_analysis_tools.capabilities.colour import display, features, metrics, transforms
from film_analysis_tools.capabilities.colour.features import (
    feature_columns,
    hue_degrees,
    luma,
    saturation,
)

__all__ = [
    "display",
    "feature_columns",
    "features",
    "hue_degrees",
    "luma",
    "metrics",
    "saturation",
    "transforms",
]
