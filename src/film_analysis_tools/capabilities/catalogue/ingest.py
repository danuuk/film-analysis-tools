"""Reading an existing survey into the catalogue's own types.

The expensive pass has already been run: 37,084 rows of ffmpeg `signalstats` + `scdet` over a
whole feature, kept on disk. This module maps that output onto :class:`FrameSurvey` and
:class:`FaceObservation` so it can be used without decoding anything again.

Two things are deliberately explicit rather than inferred.

**The column mapping is declared, not guessed.** ``SIGNALSTATS_COLUMNS`` names which producer
column becomes which survey column. A different producer is a different mapping, not a rewrite —
and a renamed column fails loudly instead of arriving as zeros. The bug that motivates this: the
CSV carries both ``time`` (empty) and ``pts_time`` (real), and reading the wrong one produced
8,660 intervals of zero duration whose every aggregate still looked plausible.

**The face probe time is reconstructed, and that reconstruction is an assumption.** The scout
recorded a verdict per scene but not the timestamp it was taken at; it probed the scene midpoint.
Every ``distance_s`` — and therefore every confidence tier in :mod:`annotate` — rests on that.
It is a parameter here so it can be corrected rather than re-derived by reading the scout.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from film_analysis_tools.capabilities.catalogue.annotate import FaceObservation
from film_analysis_tools.capabilities.catalogue.survey import REQUIRED_COLUMNS, FrameSurvey
from film_analysis_tools.capabilities.source.record import Cadence
from film_analysis_tools.core.errors import DataError

#: survey column <- producer column, for ffmpeg ``signalstats`` + ``scdet``.
SIGNALSTATS_COLUMNS: Mapping[str, str] = {
    "time_s": "pts_time",
    "motion": "mafd",
    "cut_score": "score",
    "level_mean": "yavg",
    "level_low": "ylow",
    "level_high": "yhigh",
    "level_min": "ymin",
    "level_max": "ymax",
    "saturation_mean": "satavg",
    "hue_median": "huemed",
    "bit_depth": "ybitdepth",
}


def _is_required(target: str) -> bool:
    """Derived from the survey's own contract rather than restated here, so the two cannot drift."""
    return target in REQUIRED_COLUMNS


def read_survey(
    path: Path | str,
    *,
    source_id: str,
    cadence: Cadence,
    sample_rate_hz: float,
    columns: Mapping[str, str] = SIGNALSTATS_COLUMNS,
    code_floor: float = 64.0,
    code_ceiling: float = 940.0,
    notes: Mapping[str, Any] | None = None,
) -> FrameSurvey:
    """Load a per-frame metrics CSV as a :class:`FrameSurvey`.

    A column named in the mapping but absent from the file raises, unless it is optional. A column
    present but entirely empty also raises: that is the failure mode this guards, and it is silent
    otherwise.
    """
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DataError(f"survey file is empty: {path}")
    available = set(rows[0])

    loaded: dict[str, np.ndarray] = {}
    for target, producer in columns.items():
        if producer not in available:
            if not _is_required(target):
                continue
            raise DataError(
                f"survey file {path} has no column {producer!r} (needed for {target!r}); "
                f"available: {sorted(available)}"
            )
        filled = sum(1 for row in rows if (row[producer] or "").strip())
        if filled == 0:
            if not _is_required(target):
                continue
            raise DataError(
                f"column {producer!r} (for {target!r}) is present but empty in every one of "
                f"{len(rows)} rows"
            )
        loaded[target] = np.asarray([float(row[producer] or 0.0) for row in rows], dtype=np.float64)

    return FrameSurvey(
        source_id=source_id,
        columns=loaded,
        cadence=cadence,
        sample_rate_hz=sample_rate_hz,
        code_floor=code_floor,
        code_ceiling=code_ceiling,
        notes=dict(notes or {}),
    )


def read_face_probes(
    report_path: Path | str,
    scene_catalog_path: Path | str,
    *,
    source_id: str = "",
    probe_fraction: float = 0.5,
) -> list[FaceObservation]:
    """Load per-scene face verdicts, timed at the frame the scout actually probed.

    ``probe_fraction`` is where in each scene that frame sat — 0.5 for the midpoint. This is the
    assumption the whole confidence ladder rests on: get it wrong and every ``distance_s`` is
    wrong, while nothing downstream would look amiss.

    Scenes named in the report but missing from the catalogue are skipped; there is no start time
    to place them at, and inventing one would fabricate the distance.
    """
    catalogue = json.loads(Path(scene_catalog_path).read_text())
    scenes = {scene["scene_id"]: scene for scene in catalogue["scenes"]}

    observations: list[FaceObservation] = []
    with Path(report_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            scene = scenes.get(row["scene_id"])
            if scene is None:
                continue
            start = float(scene["start_sec"])
            observations.append(
                FaceObservation(
                    time_s=start + probe_fraction * float(scene["duration_sec"]),
                    detected=(row.get("face_detected", "") or "").strip().lower() in ("true", "1"),
                    count=int(float(row.get("face_count") or 0)),
                    area_ratio=float(row.get("best_face_area_ratio") or 0.0),
                    detection_score=float(row.get("max_detection_score") or 0.0),
                    source_scene=row["scene_id"],
                    source_id=source_id,
                )
            )
    if not observations:
        raise DataError(
            f"no scene in {report_path} matched {scene_catalog_path}; the two files probably "
            "describe different sources"
        )
    return observations


__all__ = ["SIGNALSTATS_COLUMNS", "read_face_probes", "read_survey"]
