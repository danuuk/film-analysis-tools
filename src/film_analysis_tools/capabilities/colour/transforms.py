"""Analytic transforms with known behaviour.

These are not emulation models. They exist so the comparison machinery can be exercised —
and tested — against transforms whose effect is known in closed form, with no engine build,
no bundle and no footage. A harness that cannot reproduce a known answer is not evidence
about anything else.

Real emulation models arrive as ``forward/`` adapters in P8 and satisfy the same
``Transform`` protocol, so nothing above this layer changes when they do.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from film_analysis_tools.capabilities.colour.features import BT2020_LUMA_WEIGHTS, luma
from film_analysis_tools.core.errors import SelectionError
from film_analysis_tools.core.protocols import RGB, Transform


def identity() -> Transform:
    """The null transform. A comparison of identity against identity must show no effect."""

    def apply(rgb: RGB, /) -> RGB:
        return np.asarray(rgb, dtype=np.float64)

    return apply


def channel_gain(red: float = 1.0, green: float = 1.0, blue: float = 1.0) -> Transform:
    """Per-channel linear gain. Shifts hue whenever the gains differ."""
    gains = np.asarray([red, green, blue], dtype=np.float64)

    def apply(rgb: RGB, /) -> RGB:
        return np.asarray(rgb, dtype=np.float64) * gains

    return apply


def exposure(stops: float) -> Transform:
    """Uniform exposure change. Moves luma, leaves hue and relative saturation alone."""
    scale = float(2.0**stops)

    def apply(rgb: RGB, /) -> RGB:
        return np.asarray(rgb, dtype=np.float64) * scale

    return apply


def saturate(amount: float) -> Transform:
    """Scale distance from the neutral axis, holding luma. ``1.0`` is a no-op."""

    def apply(rgb: RGB, /) -> RGB:
        values = np.asarray(rgb, dtype=np.float64)
        neutral = luma(values)[..., None]
        return neutral + (values - neutral) * amount

    return apply


def tone_gamma(gamma: float) -> Transform:
    """Apply a power curve to luma, preserving channel ratios. Monotone for ``gamma > 0``."""
    if gamma <= 0.0:
        raise SelectionError(f"gamma must be positive: {gamma}")

    def apply(rgb: RGB, /) -> RGB:
        values = np.asarray(rgb, dtype=np.float64)
        before = luma(values)
        safe = np.maximum(before, 1.0e-12)
        after = safe**gamma
        return values * (after / safe)[..., None]

    return apply


def compose(*transforms: Transform) -> Transform:
    """Apply transforms left to right."""

    def apply(rgb: RGB, /) -> RGB:
        values = np.asarray(rgb, dtype=np.float64)
        for transform in transforms:
            values = transform(values)
        return values

    return apply


#: Transforms addressable by name, for the CLI and for reproducible study declarations.
BUILT_INS: Mapping[str, Transform] = {
    "identity": identity(),
    "warm_gain": channel_gain(red=1.06, blue=0.96),
    "cool_gain": channel_gain(red=0.96, blue=1.06),
    "green_gain": channel_gain(green=1.05),
    "desaturate": saturate(0.85),
    "saturate": saturate(1.15),
    "lift_shadows": tone_gamma(0.9),
    "crush_shadows": tone_gamma(1.1),
    "expose_up": exposure(0.5),
}


def named(name: str) -> Transform:
    if name not in BUILT_INS:
        raise SelectionError(f"unknown transform {name!r}; available: {sorted(BUILT_INS)}")
    return BUILT_INS[name]


__all__ = [
    "BT2020_LUMA_WEIGHTS",
    "BUILT_INS",
    "channel_gain",
    "compose",
    "exposure",
    "identity",
    "named",
    "saturate",
    "tone_gamma",
]
