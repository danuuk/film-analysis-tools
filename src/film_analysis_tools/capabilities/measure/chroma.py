"""What the material can support a claim about — enforced by type, not by caveat.

A 4:2:0 delivery master can produce a **luma-appearance** profile. It cannot produce an
authoritative colour-layer covariance profile, because the chroma planes were subsampled and
reconstructed by the decoder: any RGB residual covariance measured from them describes the
upsampler, not the film.

The legacy path knew this and said so repeatedly — "4:2:0/4:2:2 or source-y: chroma unmeasurable",
"do not treat upsampled delivery RGB residual covariance as film-grain chroma correlation",
`rgb_correlation` nulled when a source was luma-only. Those caveats were correct and well
written. They were also *strings*, sitting beside a field that still existed and could still be
read by anything that did not stop to read them.

Here the constraint is structural instead. :class:`LumaAppearanceProfile` has **no colour
covariance field at all** — a claim it cannot express is a claim it cannot leak. Constructing a
:class:`ColourGrainProfile` requires evidence of full chroma support, checked at construction, so
subsampled material cannot produce one by any route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from film_analysis_tools.core.errors import SelectionError


class ChromaSupport(Enum):
    """Whether the material can support a colour-layer claim."""

    LUMA_ONLY = "luma_only"
    """Subsampled or luma-only. Colour covariance would measure the upsampler."""

    FULL = "full"
    """4:4:4, RGB scan, or laboratory measurement. Colour is genuinely present."""

    UNKNOWN = "unknown"
    """Not established. Treated as luma-only, because assuming otherwise is the failure."""

    @property
    def supports_colour(self) -> bool:
        return self is ChromaSupport.FULL


#: Pixel-format fragments that indicate genuinely co-sited full-resolution colour.
FULL_CHROMA_MARKERS: tuple[str, ...] = ("444", "rgb", "gbr", "bgr")
#: Fragments that indicate subsampled or luma-only material.
SUBSAMPLED_MARKERS: tuple[str, ...] = ("420", "422", "411", "410", "nv12", "nv21", "gray", "grey")


def chroma_support_of(pixel_format: str) -> ChromaSupport:
    """Classify a source pixel format.

    Unrecognised formats return :attr:`ChromaSupport.UNKNOWN`, which is treated exactly like
    luma-only. Guessing in the permissive direction is how an upsampled residual becomes a
    "measured" colour correlation.
    """
    lowered = (pixel_format or "").lower()
    if not lowered:
        return ChromaSupport.UNKNOWN
    if any(marker in lowered for marker in SUBSAMPLED_MARKERS):
        return ChromaSupport.LUMA_ONLY
    if any(marker in lowered for marker in FULL_CHROMA_MARKERS):
        return ChromaSupport.FULL
    return ChromaSupport.UNKNOWN


@dataclass(frozen=True)
class LumaAppearanceProfile:
    """A grain profile a delivery master can honestly support.

    There is deliberately no colour-covariance field. This is not an oversight to be filled in
    later: it is the representation of what subsampled material can and cannot say.
    """

    source_identity: str
    chroma_support: ChromaSupport
    amplitude_points: tuple[tuple[float, float], ...] = ()
    """``(level, sigma)`` pairs — the measured amplitude-versus-level evidence."""

    measured_level_range: tuple[float, float] = (0.0, 0.0)
    notes: tuple[str, ...] = ()

    @property
    def claims_colour(self) -> bool:
        return False

    def as_record(self) -> dict[str, Any]:
        return {
            "kind": "luma_appearance",
            "source_identity": self.source_identity,
            "chroma_support": self.chroma_support.value,
            "claims_colour": False,
            "amplitude_points": [list(point) for point in self.amplitude_points],
            "measured_level_range": list(self.measured_level_range),
            "colour_limitation": (
                "Fitted from material that cannot support a colour-layer claim. Colour grain "
                "requires 4:4:4 scans, laboratory samples, or material granularity data."
            ),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ColourGrainProfile:
    """A grain profile with a genuine colour-layer claim.

    Constructible only from material with full chroma support. The check is at construction
    rather than at export, so there is no window in which an unsupported instance exists.
    """

    source_identity: str
    chroma_support: ChromaSupport
    channel_covariance: np.ndarray
    amplitude_points: tuple[tuple[float, float], ...] = ()
    measured_level_range: tuple[float, float] = (0.0, 0.0)
    evidence_basis: str = ""
    """How colour was established: ``scan_444``, ``laboratory``, ``granularity_data``."""

    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.chroma_support.supports_colour:
            raise SelectionError(
                f"a colour-grain profile cannot be built from {self.chroma_support.value} "
                "material: measured RGB residual covariance there describes chroma upsampling, "
                "not film. Use LumaAppearanceProfile, or supply 4:4:4, laboratory, or "
                "material-granularity evidence."
            )
        if not self.evidence_basis:
            raise SelectionError("a colour-grain profile must name how colour was established")
        matrix = np.asarray(self.channel_covariance, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise SelectionError(f"channel covariance must be 3x3, got {matrix.shape}")
        if not np.allclose(matrix, matrix.T, atol=1e-9):
            raise SelectionError("channel covariance must be symmetric")

    @property
    def claims_colour(self) -> bool:
        return True

    def as_record(self) -> dict[str, Any]:
        return {
            "kind": "colour_grain",
            "source_identity": self.source_identity,
            "chroma_support": self.chroma_support.value,
            "claims_colour": True,
            "evidence_basis": self.evidence_basis,
            "channel_covariance": np.asarray(self.channel_covariance).tolist(),
            "amplitude_points": [list(point) for point in self.amplitude_points],
            "measured_level_range": list(self.measured_level_range),
            "notes": list(self.notes),
        }


def profile_for(
    *,
    source_identity: str,
    pixel_format: str,
    amplitude_points: tuple[tuple[float, float], ...] = (),
    measured_level_range: tuple[float, float] = (0.0, 0.0),
    notes: tuple[str, ...] = (),
) -> LumaAppearanceProfile:
    """The profile this material can honestly support.

    Always a luma-appearance profile: a colour claim needs evidence supplied deliberately, never
    inferred from a pixel format that merely happens to carry three channels after decoding.
    """
    return LumaAppearanceProfile(
        source_identity=source_identity,
        chroma_support=chroma_support_of(pixel_format),
        amplitude_points=amplitude_points,
        measured_level_range=measured_level_range,
        notes=notes,
    )


__all__ = [
    "FULL_CHROMA_MARKERS",
    "SUBSAMPLED_MARKERS",
    "ChromaSupport",
    "ColourGrainProfile",
    "LumaAppearanceProfile",
    "chroma_support_of",
    "profile_for",
]
