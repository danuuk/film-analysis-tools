"""The interval layer: the temporal unit the catalogue is indexed on."""

from __future__ import annotations

import numpy as np
import pytest

from film_analysis_tools.capabilities.catalogue import intervals as iv
from film_analysis_tools.capabilities.catalogue import survey as sv
from film_analysis_tools.capabilities.source.record import Cadence
from film_analysis_tools.core.errors import DataError, SelectionError

RATE = 4.0


def _survey(
    seconds: float = 60.0,
    *,
    motion: float | np.ndarray = 0.5,
    level: float | np.ndarray = 300.0,
    low: float | np.ndarray | None = None,
    high: float | np.ndarray | None = None,
    cut_at: float | None = None,
    **extra: object,
) -> sv.FrameSurvey:
    count = int(seconds * RATE)
    time = np.arange(count) / RATE
    cut = np.zeros(count)
    if cut_at is not None:
        cut[int(cut_at * RATE)] = 40.0
    columns: dict[str, np.ndarray] = {
        "time_s": time,
        "motion": np.full(count, motion) if np.isscalar(motion) else np.asarray(motion),
        "cut_score": cut,
        "level_mean": np.full(count, level) if np.isscalar(level) else np.asarray(level),
    }
    if low is not None:
        columns["level_low"] = np.full(count, low) if np.isscalar(low) else np.asarray(low)
    if high is not None:
        columns["level_high"] = np.full(count, high) if np.isscalar(high) else np.asarray(high)
    for name, value in extra.items():
        columns[name] = np.full(count, value)  # type: ignore[arg-type]
    return sv.FrameSurvey(
        source_id="src", columns=columns, cadence=Cadence(24000, 1001), sample_rate_hz=RATE
    )


# ------------------------------------------------------------------- the survey


def test_a_survey_needs_its_required_columns() -> None:
    with pytest.raises(DataError, match="missing required columns"):
        sv.FrameSurvey(
            source_id="s",
            columns={"time_s": np.arange(4.0)},
            cadence=Cadence(25),
            sample_rate_hz=4.0,
        )


def test_a_constant_time_column_is_refused() -> None:
    """The bug this check exists for: a dead time column produced 8,660 intervals of zero
    duration, and every aggregate over them looked plausible while describing nothing."""
    with pytest.raises(DataError, match="must increase strictly"):
        sv.FrameSurvey(
            source_id="s",
            columns={
                "time_s": np.zeros(8),
                "motion": np.zeros(8),
                "cut_score": np.zeros(8),
                "level_mean": np.zeros(8),
            },
            cadence=Cadence(25),
            sample_rate_hz=4.0,
        )


def test_levels_normalise_across_the_declared_code_range() -> None:
    survey = _survey(level=502.0)  # midpoint of 64..940
    assert float(survey.normalised("level_mean")[0]) == pytest.approx(0.5, abs=0.001)


def test_values_beyond_the_declared_range_are_not_clipped() -> None:
    """A source exceeding its declared ceiling is saying the declaration is wrong."""
    assert float(_survey(level=1000.0).normalised("level_mean")[0]) > 1.0


# ------------------------------------------------------------------- building


def test_intervals_tile_the_survey_with_overlap() -> None:
    built = iv.build_intervals(_survey(60.0), window_s=2.0, stride_s=1.0)
    assert len(built) > 50
    assert built[0].duration_s == pytest.approx(1.75, abs=0.01)  # 8 samples at 4 Hz
    assert built[1].start_s - built[0].start_s == pytest.approx(1.0, abs=0.01)


def test_an_interval_is_short_enough_to_describe_its_own_frames() -> None:
    """The whole point of the unit. A 298-second scene aggregate cannot do this."""
    for interval in iv.build_intervals(_survey(30.0), window_s=2.0):
        assert interval.duration_s <= 2.0


def test_a_survey_shorter_than_one_interval_is_an_error() -> None:
    with pytest.raises(DataError, match="fewer than"):
        iv.build_intervals(_survey(1.0), window_s=5.0)


def test_intervals_containing_a_cut_are_dropped() -> None:
    with_cut = iv.build_intervals(_survey(20.0, cut_at=10.0), window_s=2.0, stride_s=1.0)
    without = iv.build_intervals(_survey(20.0), window_s=2.0, stride_s=1.0)
    assert len(with_cut) < len(without)
    assert all(interval.cut_free for interval in with_cut)


def test_cuts_can_be_kept_for_inspection() -> None:
    kept = iv.build_intervals(_survey(20.0, cut_at=10.0), window_s=2.0, drop_cuts=False)
    assert any(not interval.cut_free for interval in kept)


def test_a_level_ramp_is_recorded() -> None:
    count = int(20.0 * RATE)
    ramp = 200.0 + np.arange(count) * 2.0
    built = iv.build_intervals(_survey(20.0, level=ramp), window_s=2.0)
    assert built[0].level_slope > 0
    assert built[0].level_drift > 0


def test_motion_is_summarised_by_its_upper_tail() -> None:
    """One moving frame contaminates every difference it takes part in, so the p90 matters
    more than the mean."""
    count = int(10.0 * RATE)
    spiky = np.full(count, 0.1)
    spiky[::8] = 9.0
    built = iv.build_intervals(_survey(10.0, motion=spiky), window_s=2.0)
    assert built[0].motion_p90 > built[0].motion_mean * 2


# ------------------------------------------------- is versus contains a band


def test_band_asks_what_the_interval_is() -> None:
    assert iv.build_intervals(_survey(10.0, level=100.0), window_s=2.0)[0].band == "shadow"
    assert iv.build_intervals(_survey(10.0, level=900.0), window_s=2.0)[0].band == "highlight"


def test_contains_band_finds_a_lamp_in_a_dark_room() -> None:
    """The decomposition that matters.

    An interval whose frame *mean* is deep shadow can still hold a practical light. Judging by
    the mean alone is what makes a whole film read as midtone and reports zero highlights.
    """
    dark_with_highlight = iv.build_intervals(
        _survey(10.0, level=100.0, low=70.0, high=900.0), window_s=2.0
    )[0]
    assert dark_with_highlight.band == "shadow"
    assert dark_with_highlight.contains_band("highlight")
    assert dark_with_highlight.contains_band("shadow")

    uniformly_dark = iv.build_intervals(
        _survey(10.0, level=100.0, low=70.0, high=120.0), window_s=2.0
    )[0]
    assert not uniformly_dark.contains_band("highlight")


def test_an_unknown_band_is_refused() -> None:
    with pytest.raises(SelectionError, match="unknown band"):
        iv.build_intervals(_survey(10.0), window_s=2.0)[0].contains_band("chartreuse")


# --------------------------------------------------------------------- index


def test_selection_filters_on_every_stated_condition() -> None:
    count = int(60.0 * RATE)
    motion = np.where(np.arange(count) < count // 2, 0.1, 9.0)
    idx = iv.index(iv.build_intervals(_survey(60.0, motion=motion), window_s=2.0, stride_s=1.0))
    quiet = idx.select(max_motion=1.0)
    assert 0 < len(quiet) < len(idx)
    assert all(interval.motion_p90 <= 1.0 for interval in quiet)


def test_a_per_source_cap_stops_one_source_dominating() -> None:
    idx = iv.index(iv.build_intervals(_survey(60.0), window_s=2.0, stride_s=1.0))
    assert len(idx.select(per_source_cap=5)) == 5


def test_covered_time_does_not_double_count_overlap() -> None:
    idx = iv.index(iv.build_intervals(_survey(60.0), window_s=2.0, stride_s=1.0))
    assert idx.total_duration_s() <= 60.0
    assert idx.total_duration_s() > 50.0


def test_band_counts_report_both_questions() -> None:
    idx = iv.index(
        iv.build_intervals(_survey(20.0, level=100.0, low=70.0, high=900.0), window_s=2.0)
    )
    assert idx.band_counts()["shadow"] == len(idx)
    assert idx.band_counts()["highlight"] == 0
    assert idx.band_counts(contains=True)["highlight"] == len(idx)


def test_the_index_serialises_and_summarises() -> None:
    idx = iv.index(iv.build_intervals(_survey(30.0), window_s=2.0, stride_s=1.0))
    record = idx.as_record()
    for key in ("intervals", "sources", "covered_s", "band_counts", "contains_band_counts"):
        assert key in record, key
    assert "intervals over" in idx.summary()
