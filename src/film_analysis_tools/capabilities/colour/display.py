"""Encoding linear samples for display.

Sample packs hold linear Rec.709 at L100. Showing a swatch on screen needs the sRGB transfer
function applied, otherwise every patch looks far too dark and a report misrepresents what the
transform did.

This is a display convenience, not a colour-management claim: it applies the sRGB OETF to
clamped linear values and stops there. No white-point adaptation, no gamut mapping, no tone
mapping. Out-of-range values are clipped, and :func:`clipped_fraction` reports how much was
clipped so a report can say so rather than hiding it.
"""

from __future__ import annotations

import numpy as np

from film_analysis_tools.core.protocols import RGB

SRGB_LINEAR_CUTOFF = 0.0031308


def srgb_encode(linear: RGB) -> np.ndarray:
    """Apply the sRGB opto-electronic transfer function to clamped linear values."""
    values = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    low = values * 12.92
    high = 1.055 * np.power(values, 1.0 / 2.4) - 0.055
    return np.where(values <= SRGB_LINEAR_CUTOFF, low, high)


def clipped_fraction(linear: RGB) -> float:
    """Share of channel values outside ``[0, 1]`` — what the swatches cannot show."""
    values = np.asarray(linear, dtype=np.float64)
    if values.size == 0:
        return 0.0
    outside = np.count_nonzero((values < 0.0) | (values > 1.0))
    return float(outside) / float(values.size)


def hex_colours(linear: RGB) -> list[str]:
    """Encode an ``(N, 3)`` block of linear samples as ``#rrggbb`` strings."""
    encoded = srgb_encode(np.atleast_2d(np.asarray(linear, dtype=np.float64)))
    bytes_ = np.rint(encoded * 255.0).astype(np.int64).clip(0, 255)
    return [f"#{row[0]:02x}{row[1]:02x}{row[2]:02x}" for row in bytes_]


__all__ = ["SRGB_LINEAR_CUTOFF", "clipped_fraction", "hex_colours", "srgb_encode"]
