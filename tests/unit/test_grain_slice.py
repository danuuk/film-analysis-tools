"""The vertical slice: the parts that can be checked without the private sources.

The ffmpeg stages are exercised by running the study over real material, not here. What is worth
pinning down in a test is the reasoning the run depends on — which domain each question is asked
in, and whether coverage is reported honestly.
"""

from __future__ import annotations

import numpy as np
import pytest

from film_analysis_tools.capabilities.measure import admissibility
from film_analysis_tools.core.errors import DataError
from film_analysis_tools.studies import grain_slice as gs

PQ_STREAM = {"width": 3840, "height": 2160, "color_range": "tv", "r_frame_rate": "24000/1001"}
SLOG_STREAM = {"width": 1920, "height": 1080, "color_range": "pc", "r_frame_rate": "24000/1001"}


def _plan(**kwargs: object) -> gs.SourcePlan:
    base: dict[str, object] = {
        "source_id": "s",
        "path": gs.Path("/nowhere.mkv"),
        "edition": "e",
        "transfer": "pq",
    }
    base.update(kwargs)
    return gs.SourcePlan(**base)  # type: ignore[arg-type]


def test_an_unknown_transfer_is_refused() -> None:
    with pytest.raises(DataError, match="unsupported transfer"):
        _plan(transfer="rec709")


# ------------------------------------------------------------- domains


def test_admissibility_is_asked_in_the_container_domain() -> None:
    """The defect that stopped the whole film measuring.

    Shadow codes survive as ordinary small numbers in the container, but a PQ EOTF compresses
    them until they underflow to zero. Asking about clipping after the transfer reported 87.6%
    of an ordinary film interval as clipped black, and 100% of the frame as carrying no noise.
    """
    rng = np.random.default_rng(0)
    container = np.clip(0.10 + rng.normal(0.0, 0.004, (8, 128, 128)), 0.0, 1.0)

    in_container = admissibility.clipping_evidence(container, ceiling=1.0, floor=0.0)
    linear = gs.to_linear(_plan(), container, PQ_STREAM)
    after_transfer = admissibility.clipping_evidence(linear, ceiling=1.0, floor=0.0)

    assert not in_container.is_clipped
    assert after_transfer.total_fraction > in_container.total_fraction
    assert linear.max() < 0.02, "PQ code 0.10 is a very small amount of light"


def test_levels_land_on_a_common_scale_whatever_the_transfer() -> None:
    """1.0 must mean diffuse white for every source, or no two are comparable (gap 8)."""
    pq_white = gs.to_linear(_plan(transfer="pq"), np.full((2, 8, 8), 0.5081), PQ_STREAM)
    slog_white = gs.to_linear(_plan(transfer="slog3"), np.full((2, 8, 8), 0.5977), SLOG_STREAM)
    assert pq_white.mean() == pytest.approx(1.0, rel=0.15)
    assert slog_white.mean() == pytest.approx(1.0, rel=0.15)


def test_limited_and_full_range_are_decoded_differently() -> None:
    mid = np.full((2, 8, 8), 0.5)
    limited = gs.to_linear(_plan(transfer="pq"), mid, PQ_STREAM)
    full = gs.to_linear(_plan(transfer="pq"), mid, {**PQ_STREAM, "color_range": "pc"})
    assert not np.isclose(limited.mean(), full.mean())


# ------------------------------------------------------------- coverage


def test_a_stage_reports_what_it_dropped_and_why() -> None:
    stage = gs.StageCount(
        stage="windows", considered=540, accepted=245, rejected={"motion above gate": 202}
    )
    assert stage.survival == pytest.approx(245 / 540)
    assert "245" in stage.line() and "540" in stage.line()
    assert "motion above gate 202" in stage.line()
    assert stage.as_record()["survival"] == pytest.approx(0.4537, abs=1e-3)


def test_a_stage_that_saw_nothing_does_not_divide_by_zero() -> None:
    assert gs.StageCount(stage="x", considered=0, accepted=0).survival == 0.0


def test_the_slice_refuses_to_run_on_one_source() -> None:
    """Screen anchoring is the question a single source cannot answer at all."""
    with pytest.raises(DataError, match="at least two unrelated sources"):
        gs.run([_plan()])


# ------------------------------------------------------------- reporting


def _outcome(**kwargs: object) -> gs.SourceOutcome:
    plan = _plan()
    record = gs.source_record(
        gs.SourcePlan(
            source_id="s", path=gs.Path(__file__), edition="e", transfer="pq", sha256="a" * 64
        ),
        PQ_STREAM,
        gs.Crop(x=0, y=263, width=3840, height=1634),
    )
    base: dict[str, object] = {
        "plan": plan,
        "record": record,
        "stages": (gs.StageCount(stage="intervals", considered=10, accepted=3),),
        "regions": gs.rg.index([]),
        "frames_measured": 0,
        "amplitude": None,
        "spectrum": None,
        "distribution": None,
        "heterogeneity": None,
        "temporal": None,
    }
    base.update(kwargs)
    return gs.SourceOutcome(**base)  # type: ignore[arg-type]


def test_a_source_that_measured_nothing_says_so_plainly() -> None:
    text = gs.report(gs.StudyResult(outcomes=(_outcome(),)))
    assert "NONE — nothing survived the chain" in text
    assert "intervals" in text


def test_the_report_names_the_active_picture() -> None:
    """Letterbox bars sit at code 0 and read as hard-clipped black across a quarter of the
    frame; the crop that removes them has to be visible in the output."""
    text = gs.report(gs.StudyResult(outcomes=(_outcome(),)))
    assert "3840x1634" in text
    assert "letterbox cropped 263px" in text


def test_the_record_carries_provenance_and_coverage() -> None:
    record = gs.StudyResult(outcomes=(_outcome(),)).as_record()
    assert record["study"] == "grain_slice"
    assert "diffuse white" in record["level_scale"]
    source = record["sources"][0]
    assert source["source"]["sha256"] == "a" * 64
    assert source["source"]["active_picture"] == [0, 263, 3840, 1634]
    assert source["coverage"][0]["considered"] == 10
    assert source["evidence"]["amplitude"] is None
