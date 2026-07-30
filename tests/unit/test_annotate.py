"""Face and colour annotation, and the evidential status attached to each."""

from __future__ import annotations

import numpy as np
import pytest

from film_analysis_tools.capabilities.catalogue import annotate as an
from film_analysis_tools.capabilities.catalogue import intervals as iv
from film_analysis_tools.capabilities.catalogue import survey as sv
from film_analysis_tools.capabilities.source.record import Cadence
from film_analysis_tools.core.errors import SelectionError

RATE = 4.0


def _intervals(seconds: float = 40.0, *, saturation: float = 17.0) -> list[iv.Interval]:
    count = int(seconds * RATE)
    survey = sv.FrameSurvey(
        source_id="src",
        columns={
            "time_s": np.arange(count) / RATE,
            "motion": np.full(count, 0.3),
            "cut_score": np.zeros(count),
            "level_mean": np.full(count, 300.0),
            "level_low": np.full(count, 80.0),
            "level_high": np.full(count, 800.0),
            "saturation_mean": np.full(count, saturation),
            "hue_median": np.full(count, 130.0),
        },
        cadence=Cadence(24000, 1001),
        sample_rate_hz=RATE,
    )
    return iv.build_intervals(survey, window_s=2.0, stride_s=1.0)


def _probe(time_s: float, *, detected: bool = True, area: float = 0.1) -> an.FaceObservation:
    return an.FaceObservation(
        time_s=time_s, detected=detected, count=1, area_ratio=area, detection_score=0.9
    )


# ------------------------------------------------------------------ colour


def test_colour_is_measured_from_the_intervals_own_frames() -> None:
    """Unlike faces, the survey sampled every frame, so this is measurement not inheritance."""
    annotated = an.annotate(_intervals(saturation=40.0))
    assert annotated[0].colour.saturation_band == "saturated"
    assert annotated[0].colour.is_saturated


@pytest.mark.parametrize(
    ("value", "band"),
    [(5.0, "neutral"), (17.0, "low"), (26.0, "moderate"), (45.0, "saturated")],
)
def test_saturation_bands_follow_the_measured_distribution(value: float, band: str) -> None:
    assert an.saturation_band(value) == band


def test_saturation_uses_the_upper_percentile_not_the_mean() -> None:
    """The question is whether the interval *contains* saturated content; a mean over a mostly
    neutral frame hides a single vivid object."""
    interval = _intervals(saturation=17.0)[0]
    assert an.colour_annotation(interval).saturation_p90 == interval.saturation_p90


# -------------------------------------------------------- face confidence tiers


def test_a_probe_inside_the_interval_is_an_observation() -> None:
    intervals = _intervals()
    target = intervals[5]
    annotated = an.annotate([target], [_probe(target.start_s + 0.5)])[0]
    assert annotated.face.confidence is an.FaceConfidence.OBSERVED
    assert annotated.face.distance_s == 0.0
    assert annotated.face.confidence.usable_for_skin


def test_a_probe_just_outside_is_near() -> None:
    target = _intervals()[5]
    annotated = an.annotate([target], [_probe(target.end_s + 1.0)])[0]
    assert annotated.face.confidence is an.FaceConfidence.NEAR
    assert annotated.face.confidence.usable_for_skin


def test_a_distant_probe_is_inherited_and_not_acted_on() -> None:
    """The failure this tier exists for.

    Face evidence came from one probe per scene, at the midpoint. The longest scene is 878 s, so
    an interval can sit 439 s from the only frame ever checked. That is evidence a face appeared
    somewhere in the same badly-segmented range, not evidence about this interval.
    """
    target = _intervals()[2]
    annotated = an.annotate([target], [_probe(target.end_s + 300.0)])[0]
    assert annotated.face.detected
    assert annotated.face.confidence is an.FaceConfidence.INHERITED
    assert not annotated.face.confidence.usable_for_skin


def test_no_probe_means_no_face_evidence() -> None:
    annotated = an.annotate(_intervals(), [])[0]
    assert not annotated.face.detected
    assert annotated.face.confidence is an.FaceConfidence.NONE


def test_a_probe_reporting_no_face_is_not_evidence_of_one() -> None:
    target = _intervals()[5]
    annotated = an.annotate([target], [_probe(target.start_s + 0.5, detected=False)])[0]
    assert not annotated.face.detected
    assert annotated.face.confidence is an.FaceConfidence.NONE


def test_the_nearest_probe_wins() -> None:
    target = _intervals()[5]
    annotated = an.annotate(
        [target], [_probe(target.start_s - 100.0, area=0.9), _probe(target.end_s + 1.0, area=0.2)]
    )[0]
    assert annotated.face.area_ratio == pytest.approx(0.2)
    assert annotated.face.confidence is an.FaceConfidence.NEAR


# ------------------------------------------------------------------- queries


def _index(**kwargs: object) -> an.AnnotatedIndex:
    intervals = _intervals(**kwargs)  # type: ignore[arg-type]
    probes = [_probe(intervals[3].start_s + 0.5), _probe(intervals[20].start_s + 0.5, area=0.02)]
    return an.index(an.annotate(intervals, probes))


def test_face_confidence_is_a_query_dimension() -> None:
    """Asking for skin material without stating confidence is how an interval minutes from the
    only checked frame ends up in a skin corpus."""
    index = _index()
    any_face = index.select(has_face=True)
    observed = index.select(has_face=True, face_confidence=an.FaceConfidence.OBSERVED)
    assert len(observed) < len(any_face)
    assert all(e.face.confidence is an.FaceConfidence.OBSERVED for e in observed)


def test_face_area_filters_out_faces_too_small_to_sample() -> None:
    index = _index()
    assert len(index.select(has_face=True, min_face_area=0.05)) < len(index.select(has_face=True))


def test_saturation_is_a_query_dimension() -> None:
    assert _index(saturation=45.0).select(saturation="saturated")
    assert not _index(saturation=45.0).select(saturation="neutral")


def test_an_unknown_saturation_band_is_refused() -> None:
    with pytest.raises(SelectionError, match="unknown saturation band"):
        _index().select(saturation="fuchsia")


def test_interval_conditions_still_apply_after_annotation() -> None:
    index = _index()
    assert index.select(max_motion=0.1) == []
    assert index.select(max_motion=1.0)
    assert index.select(contains_band="highlight")


def test_a_per_source_cap_survives_annotation() -> None:
    assert len(_index().select(per_source_cap=4)) == 4


# ------------------------------------------------------------------ reporting


def test_the_index_reports_how_face_evidence_was_obtained() -> None:
    counts = _index().face_confidence_counts()
    assert set(counts) == {tier.value for tier in an.FaceConfidence}
    assert sum(counts.values()) == len(_index())


def test_the_summary_separates_usable_from_inherited() -> None:
    summary = _index().summary()
    assert "usable for skin" in summary
    assert "inherited evidence is not acted on" in summary


def test_an_annotated_interval_serialises_with_both_annotations() -> None:
    record = _index().intervals[3].as_record()
    assert "face" in record and "colour" in record
    assert record["face"]["confidence"]
    assert "saturation_band" in record["colour"]
    assert record["source_id"] == "src"


def test_observations_can_be_built_from_parallel_columns() -> None:
    built = an.observations_from_probes(
        [1.0, 2.0], [True, False], area_ratios=[0.2, 0.0], scenes=["a", "b"]
    )
    assert len(built) == 2
    assert built[0].detected and built[0].source_scene == "a"
    with pytest.raises(SelectionError, match="same length"):
        an.observations_from_probes([1.0], [True, False])
