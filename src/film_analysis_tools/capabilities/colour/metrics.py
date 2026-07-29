"""Per-sample metrics: what changed between two renderings of the same samples.

Each metric maps a paired ``(before, after)`` block of RGB to one value per row. Comparison
then summarises that distribution — it never collapses it to a verdict, because a pass/fail
is the shape that hides a marginal mechanism.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from film_analysis_tools.capabilities.colour import features
from film_analysis_tools.core.errors import SelectionError
from film_analysis_tools.core.protocols import RGB

Metric = Callable[[RGB, RGB], np.ndarray]


def hue_drift(before: RGB, after: RGB) -> np.ndarray:
    """Signed hue change in degrees, wrapped to ``(-180, 180]``.

    Positive is counter-clockwise on the hue circle. Wrapping matters: a shift from 359° to
    1° is +2°, not -358°.
    """
    delta = features.hue_degrees(after) - features.hue_degrees(before)
    return (delta + 180.0) % 360.0 - 180.0


def luma_ratio(before: RGB, after: RGB) -> np.ndarray:
    """Luma change as a ratio, in stops. Zero means unchanged."""
    start = np.maximum(features.luma(before), 1.0e-12)
    end = np.maximum(features.luma(after), 1.0e-12)
    return np.log2(end / start)


def chroma_delta(before: RGB, after: RGB) -> np.ndarray:
    """Absolute change in opponent chroma. Positive means more saturated."""
    return features.chroma(after) - features.chroma(before)


def saturation_delta(before: RGB, after: RGB) -> np.ndarray:
    """Change in relative saturation."""
    return features.saturation(after) - features.saturation(before)


def max_channel_delta(before: RGB, after: RGB) -> np.ndarray:
    """Largest absolute per-channel change — a blunt magnitude of the move."""
    difference = np.abs(np.asarray(after, dtype=np.float64) - np.asarray(before, dtype=np.float64))
    return np.asarray(difference.max(axis=-1))


#: Metrics addressable by name, with the unit used when reporting them.
BUILT_INS: Mapping[str, Metric] = {
    "hue_drift": hue_drift,
    "luma_ratio": luma_ratio,
    "chroma_delta": chroma_delta,
    "saturation_delta": saturation_delta,
    "max_channel_delta": max_channel_delta,
}

UNITS: Mapping[str, str] = {
    "hue_drift": "deg",
    "luma_ratio": "stops",
    "chroma_delta": "",
    "saturation_delta": "",
    "max_channel_delta": "",
}


def named(name: str) -> Metric:
    if name not in BUILT_INS:
        raise SelectionError(f"unknown metric {name!r}; available: {sorted(BUILT_INS)}")
    return BUILT_INS[name]


__all__ = [
    "BUILT_INS",
    "UNITS",
    "Metric",
    "chroma_delta",
    "hue_drift",
    "luma_ratio",
    "max_channel_delta",
    "named",
    "saturation_delta",
]
