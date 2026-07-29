"""Colour features: the substrate cohort selection and metrics are built on.

Reimplements the clean part of the legacy feature extractor — the module that was already
only ~20% verification vocabulary, against 54% for the sample-pack builder around it. The
maths is unchanged; the ceremony is not carried over.

Opponent axes are ``u = R - G`` and ``v = B - G``, with BT.2020 luma weights, matching the
existing sample packs so their stored columns and anything recomputed here agree.
"""

from __future__ import annotations

import numpy as np

from film_analysis_tools.core.protocols import RGB

BT2020_LUMA_WEIGHTS = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float64)
EPS = 1.0e-12

#: Named hue sectors, as half-open degree ranges. A sector below the neutral saturation
#: threshold is reported as ``"neutral"`` regardless of hue, because hue is meaningless there.
HUE_SECTORS: tuple[tuple[str, float, float], ...] = (
    ("red", 345.0, 15.0),
    ("orange", 15.0, 45.0),
    ("yellow", 45.0, 75.0),
    ("green", 75.0, 165.0),
    ("cyan", 165.0, 195.0),
    ("blue", 195.0, 255.0),
    ("purple", 255.0, 285.0),
    ("magenta", 285.0, 345.0),
)
NEUTRAL_SATURATION = 0.08


def luma(rgb: RGB) -> np.ndarray:
    """BT.2020 luma."""
    return np.tensordot(np.asarray(rgb, dtype=np.float64), BT2020_LUMA_WEIGHTS, axes=([-1], [0]))


def opponent_uv(rgb: RGB) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(rgb, dtype=np.float64)
    return values[..., 0] - values[..., 1], values[..., 2] - values[..., 1]


def chroma(rgb: RGB) -> np.ndarray:
    """Absolute opponent chroma — distance from the neutral axis."""
    u, v = opponent_uv(rgb)
    return np.sqrt(u * u + v * v)


def hue_degrees(rgb: RGB) -> np.ndarray:
    """Hue angle in degrees, from the standard RGB hexcone."""
    values = np.asarray(rgb, dtype=np.float64)
    high = values.max(axis=-1)
    low = values.min(axis=-1)
    span = high - low
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]

    hue = np.zeros(high.shape, dtype=np.float64)
    safe = span > EPS
    is_red = safe & (high == red)
    is_green = safe & (high == green) & ~is_red
    is_blue = safe & (high == blue) & ~is_red & ~is_green

    with np.errstate(invalid="ignore", divide="ignore"):
        hue[is_red] = ((green[is_red] - blue[is_red]) / span[is_red]) % 6.0
        hue[is_green] = (blue[is_green] - red[is_green]) / span[is_green] + 2.0
        hue[is_blue] = (red[is_blue] - green[is_blue]) / span[is_blue] + 4.0
    return (hue * 60.0) % 360.0


def saturation(rgb: RGB) -> np.ndarray:
    """Relative saturation — ``(max - min) / max``, zero where the sample is black."""
    values = np.asarray(rgb, dtype=np.float64)
    high = values.max(axis=-1)
    low = values.min(axis=-1)
    result = np.zeros(high.shape, dtype=np.float64)
    lit = high > EPS
    result[lit] = (high[lit] - low[lit]) / high[lit]
    return result


def hue_sector_names(hue_deg: np.ndarray, relative_saturation: np.ndarray) -> np.ndarray:
    """Map hue angles to named sectors, with low-saturation samples labelled ``neutral``."""
    angles = np.asarray(hue_deg, dtype=np.float64) % 360.0
    names = np.full(angles.shape, "neutral", dtype="<U8")
    for name, start, end in HUE_SECTORS:
        within = (
            (angles >= start) | (angles < end)
            if start > end
            else (angles >= start) & (angles < end)
        )
        names[within] = name
    names[np.asarray(relative_saturation, dtype=np.float64) < NEUTRAL_SATURATION] = "neutral"
    return names


def feature_columns(rgb: RGB) -> dict[str, np.ndarray]:
    """Every derived feature for a block of samples, as named columns."""
    values = np.asarray(rgb, dtype=np.float64)
    u, v = opponent_uv(values)
    hue = hue_degrees(values)
    sat = saturation(values)
    return {
        "luma_bt2020": luma(values),
        "hue_deg": hue,
        "relative_saturation": sat,
        "absolute_opponent_chroma": np.sqrt(u * u + v * v),
        "opponent_u": u,
        "opponent_v": v,
        "hue_sector": hue_sector_names(hue, sat),
    }


__all__ = [
    "BT2020_LUMA_WEIGHTS",
    "HUE_SECTORS",
    "NEUTRAL_SATURATION",
    "chroma",
    "feature_columns",
    "hue_degrees",
    "hue_sector_names",
    "luma",
    "opponent_uv",
    "saturation",
]
