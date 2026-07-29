"""Sony S-Log3 / S-Gamut3.Cine decoding.

Camera material arrives as 10-bit S-Log3 in S-Gamut3.Cine. Analysis needs linear Rec.709, so
this is the boundary between what the camera recorded and what the statistics measure.

**Evidence class: published formulae, not measured.** The transfer curve and the primaries
matrix are Sony's published definitions, transcribed and checked against their own anchors
(18% grey at code 420, 90% white at code 598). They are not derived from this camera, so a
per-unit calibration would supersede them.

Two consequences worth knowing before reading any statistic built on this:

* S-Log3 code never reaches full range. On this material the top code sits near 0.72 to 0.76,
  so a "near maximum code" test for clipped highlights detects nothing. Overexposure shows up
  in *linear* terms, above 100% reflectance.
* S-Gamut3.Cine is wider than Rec.709, so the matrix legitimately produces **negative**
  channel values for saturated sources. That is information — the colour is outside Rec.709 —
  not an error, and it breaks any naive ``(max-min)/max`` saturation measure.
"""

from __future__ import annotations

import numpy as np

from film_analysis_tools.core.protocols import RGB

#: Code value where the S-Log3 curve switches from its linear toe to the log segment.
SLOG3_BREAK_CODE = 171.2102946929
SLOG3_BREAK = SLOG3_BREAK_CODE / 1023.0

#: S-Gamut3.Cine to Rec.709, Sony published. Rows sum to 1, so neutrals stay neutral.
SGAMUT3CINE_TO_REC709 = np.asarray(
    [
        [1.6269, -0.3441, -0.2828],
        [-0.1721, 1.3604, -0.1883],
        [-0.0215, -0.0784, 1.0999],
    ],
    dtype=np.float64,
)

#: Anchors used to verify the curve, as (code value, expected linear reflectance).
CURVE_ANCHORS: tuple[tuple[float, float], ...] = ((420.0 / 1023.0, 0.18), (598.0 / 1023.0, 0.90))


def slog3_to_linear(code: RGB) -> np.ndarray:
    """S-Log3 code values in ``[0, 1]`` to scene-linear reflectance, where 0.18 is 18% grey."""
    values = np.asarray(code, dtype=np.float64)
    log_segment = (10.0 ** ((values * 1023.0 - 420.0) / 261.5)) * (0.18 + 0.01) - 0.01
    toe = (values * 1023.0 - 95.0) * 0.01125 / (SLOG3_BREAK_CODE - 95.0)
    return np.where(values >= SLOG3_BREAK, log_segment, toe)


def linear_to_slog3(linear: RGB) -> np.ndarray:
    """Inverse of :func:`slog3_to_linear`, for round-trip checks."""
    values = np.asarray(linear, dtype=np.float64)
    toe_limit = 0.01125
    log_segment = (420.0 + np.log10((np.maximum(values, -0.00999) + 0.01) / 0.19) * 261.5) / 1023.0
    toe = (values * (SLOG3_BREAK_CODE - 95.0) / toe_limit + 95.0) / 1023.0
    return np.where(values >= toe_limit, log_segment, toe)


def sgamut3cine_to_rec709(linear: RGB) -> np.ndarray:
    """Rotate S-Gamut3.Cine primaries to Rec.709. Negative outputs mean out-of-gamut colour."""
    return np.asarray(linear, dtype=np.float64) @ SGAMUT3CINE_TO_REC709.T


def decode(code: RGB) -> np.ndarray:
    """Full path: S-Log3 / S-Gamut3.Cine code values to linear Rec.709."""
    return sgamut3cine_to_rec709(slog3_to_linear(code))


def out_of_gamut_mask(linear_rec709: RGB, tolerance: float = 0.002) -> np.ndarray:
    """Samples with a channel below zero — outside Rec.709 rather than merely dark."""
    return np.any(np.asarray(linear_rec709, dtype=np.float64) < -abs(tolerance), axis=-1)


__all__ = [
    "CURVE_ANCHORS",
    "SGAMUT3CINE_TO_REC709",
    "SLOG3_BREAK",
    "SLOG3_BREAK_CODE",
    "decode",
    "linear_to_slog3",
    "out_of_gamut_mask",
    "sgamut3cine_to_rec709",
    "slog3_to_linear",
]
