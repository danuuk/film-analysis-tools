"""Reading an existing survey in, and failing loudly when a column is not what it claims."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from film_analysis_tools.capabilities.catalogue import ingest
from film_analysis_tools.capabilities.source.record import Cadence
from film_analysis_tools.core.errors import DataError

HEADER = "pts_time,mafd,score,yavg,ylow,yhigh,satavg,huemed"


def _csv(tmp_path: Path, rows: list[str], header: str = HEADER) -> Path:
    path = tmp_path / "metrics.csv"
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def _survey(path: Path) -> ingest.FrameSurvey:
    return ingest.read_survey(path, source_id="s", cadence=Cadence(24000, 1001), sample_rate_hz=4.0)


def test_a_survey_loads_with_its_declared_columns(tmp_path: Path) -> None:
    survey = _survey(
        _csv(tmp_path, ["0.0,0.5,0,300,80,800,15,120", "0.25,0.6,0,310,80,810,16,121"])
    )
    assert len(survey) == 2
    assert survey.has("saturation_mean")
    assert float(survey.column("level_mean")[1]) == 310.0


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="empty"):
        _survey(_csv(tmp_path, []))


def test_a_missing_required_column_is_named(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="no column 'pts_time'"):
        _survey(_csv(tmp_path, ["0.5,0,300"], header="mafd,score,yavg"))


def test_a_present_but_empty_column_is_refused(tmp_path: Path) -> None:
    """The bug this exists for.

    The real file carries both ``time`` (empty) and ``pts_time`` (real). Reading the empty one
    produced 8,660 intervals of zero duration, and every aggregate over them looked plausible.
    """
    with pytest.raises(DataError, match="present but empty"):
        _survey(_csv(tmp_path, ["0.0,,0,300,80,800,15,120", "0.25,,0,310,80,810,16,121"]))


def test_an_optional_column_may_be_absent(tmp_path: Path) -> None:
    survey = _survey(
        _csv(tmp_path, ["0.0,0.5,0,300", "0.25,0.6,0,310"], "pts_time,mafd,score,yavg")
    )
    assert not survey.has("saturation_mean")
    assert len(survey) == 2


# ------------------------------------------------------------------ face probes


def _scout(tmp_path: Path, *, scene_id: str = "s001") -> tuple[Path, Path]:
    report = tmp_path / "face_scout_report.csv"
    report.write_text(
        "scene_id,face_detected,face_count,best_face_area_ratio,max_detection_score\n"
        f"{scene_id},true,1,0.08,0.92\n"
    )
    catalogue = tmp_path / "scene_catalog.json"
    catalogue.write_text(
        json.dumps({"scenes": [{"scene_id": "s001", "start_sec": 100.0, "duration_sec": 20.0}]})
    )
    return report, catalogue


def test_the_probe_is_placed_where_the_scout_looked(tmp_path: Path) -> None:
    report, catalogue = _scout(tmp_path)
    observation = ingest.read_face_probes(report, catalogue)[0]
    assert observation.time_s == pytest.approx(110.0)  # midpoint of 100..120
    assert observation.detected and observation.area_ratio == pytest.approx(0.08)


def test_the_probe_position_is_a_correctable_assumption(tmp_path: Path) -> None:
    """Every confidence tier rests on this; getting it wrong is invisible downstream."""
    report, catalogue = _scout(tmp_path)
    assert ingest.read_face_probes(report, catalogue, probe_fraction=0.0)[0].time_s == 100.0


def test_a_scene_with_no_start_time_is_skipped_not_invented(tmp_path: Path) -> None:
    report, catalogue = _scout(tmp_path, scene_id="ghost")
    with pytest.raises(DataError, match="different sources"):
        ingest.read_face_probes(report, catalogue)
