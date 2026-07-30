"""The cheap per-frame survey: the raw material every interval is built from.

One reduced-resolution pass over a source produces a per-frame record of motion, level,
saturation, hue and bit depth. It is cheap enough to run over a whole feature and its output is
small enough to keep, so every later question about *which* material to use is answered by
re-reading it rather than by decoding again.

**The survey selects; it does not measure.** It is downscaled and pre-blurred, which destroys
exactly the high-frequency detail grain lives in. Nothing here can see grain, and no statistic
computed from it should ever be presented as a grain measurement. Its job is to say where to look.

Stored columnar — ``dict[str, np.ndarray]`` — for the same reason the sample table is: the data is
small, masking is instant, and the representation converts to anything later without lock-in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from film_analysis_tools.capabilities.source.record import Cadence
from film_analysis_tools.core.errors import DataError, SelectionError

#: Columns a survey must carry. Anything else is optional and passes through.
REQUIRED_COLUMNS: tuple[str, ...] = ("time_s", "motion", "cut_score", "level_mean")

#: Columns used when present, ignored when absent.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "level_low",
    "level_high",
    "level_min",
    "level_max",
    "saturation_mean",
    "hue_median",
    "bit_depth",
)


@dataclass(frozen=True)
class FrameSurvey:
    """Per-frame metrics over one source, with the encoding they are expressed in."""

    source_id: str
    columns: Mapping[str, np.ndarray]
    cadence: Cadence
    """Cadence of the *source*, not of the survey — the survey is subsampled in time."""

    sample_rate_hz: float
    """How often the survey sampled. Below the source cadence by design."""

    code_floor: float = 64.0
    code_ceiling: float = 940.0
    """The code range levels are expressed on. Recorded so normalisation is reversible and so a
    full-range source is never silently read as legal-range."""

    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [name for name in REQUIRED_COLUMNS if name not in self.columns]
        if missing:
            raise DataError(f"survey is missing required columns: {missing}")
        lengths = {int(value.shape[0]) for value in self.columns.values()}
        if len(lengths) > 1:
            raise DataError(f"survey columns have differing lengths: {sorted(lengths)}")
        if self.code_ceiling <= self.code_floor:
            raise SelectionError(f"code range is inverted: {self.code_floor}..{self.code_ceiling}")
        # A constant or non-monotonic time column silently produces intervals of zero duration,
        # and every downstream aggregate then looks plausible while describing nothing.
        times = np.asarray(self.columns["time_s"], dtype=np.float64)
        if times.size > 1 and not np.all(np.diff(times) > 0):
            raise DataError(
                "survey time_s must increase strictly; got a constant or non-monotonic column "
                f"(first {times[:3].tolist()}, last {times[-2:].tolist()})"
            )
        if self.sample_rate_hz <= 0:
            raise SelectionError(f"sample rate must be positive: {self.sample_rate_hz}")

    def __len__(self) -> int:
        for value in self.columns.values():
            return int(value.shape[0])
        return 0

    @property
    def duration_s(self) -> float:
        times = self.columns["time_s"]
        return float(times[-1] - times[0]) if times.size else 0.0

    @property
    def column_names(self) -> list[str]:
        return sorted(self.columns)

    def column(self, name: str) -> np.ndarray:
        if name not in self.columns:
            raise DataError(f"survey has no column {name!r}; available: {self.column_names}")
        return self.columns[name]

    def has(self, name: str) -> bool:
        return name in self.columns

    def normalised(self, name: str) -> np.ndarray:
        """A level column mapped onto 0..1 across the declared code range.

        Values outside the range are kept rather than clipped: a source that exceeds its declared
        ceiling is telling you the declaration is wrong, and clipping would hide that.
        """
        span = self.code_ceiling - self.code_floor
        return (self.column(name).astype(np.float64) - self.code_floor) / span

    def as_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "frames": len(self),
            "duration_s": self.duration_s,
            "cadence": str(self.cadence),
            "sample_rate_hz": self.sample_rate_hz,
            "code_floor": self.code_floor,
            "code_ceiling": self.code_ceiling,
            "columns": self.column_names,
            "notes": dict(self.notes),
        }


__all__ = ["OPTIONAL_COLUMNS", "REQUIRED_COLUMNS", "FrameSurvey"]
