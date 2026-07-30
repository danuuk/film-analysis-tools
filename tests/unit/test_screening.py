"""Per-measurement scene screening."""

from __future__ import annotations

import pytest

from film_analysis_tools.capabilities.measure import screening as scr


def _scene(**kwargs: object) -> scr.SceneCandidate:
    base: dict[str, object] = {
        "scene_id": "s001",
        "duration_s": 8.0,
        "frame_count": 120,
        "static_score": 0.8,
        "cut_score": 0.1,
        "signal_bit_depth": 10.0,
        "peak_code": 0.8,
        "floor_code": 0.05,
        "median_level": 0.1,
        "level_p10": 0.01,
        "level_p90": 0.4,
    }
    base.update(kwargs)
    return scr.SceneCandidate(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------- per-measurement


def test_a_clean_scene_is_admissible_for_every_measurement() -> None:
    for measurement in scr.Measurement:
        assert scr.screen_scene(_scene(), measurement).admissible, measurement


def test_requirements_differ_by_measurement() -> None:
    """A scene admissible for one may be useless for another — the differences are the point."""
    assert scr.REQUIREMENTS[scr.Measurement.SPECTRUM] != scr.REQUIREMENTS[scr.Measurement.AMPLITUDE]
    assert "A8" in scr.REQUIREMENTS[scr.Measurement.SPECTRUM]
    assert "A6" in scr.REQUIREMENTS[scr.Measurement.DISTRIBUTION]
    assert (
        scr.REQUIREMENTS[scr.Measurement.HETEROGENEITY]
        < scr.REQUIREMENTS[scr.Measurement.AMPLITUDE]
    )


def test_resampling_disqualifies_only_the_spectrum() -> None:
    """Scaling correlates noise spatially, so the measured spectrum would be the scaler's."""
    scene = _scene(resampled=True)
    assert not scr.screen_scene(scene, scr.Measurement.SPECTRUM).admissible
    assert scr.screen_scene(scene, scr.Measurement.TEMPORAL).admissible


def test_clipping_disqualifies_distribution_but_not_temporal() -> None:
    scene = _scene(peak_code=0.9999)
    assert not scr.screen_scene(scene, scr.Measurement.DISTRIBUTION).admissible
    assert scr.screen_scene(scene, scr.Measurement.TEMPORAL).admissible


@pytest.mark.parametrize(
    ("change", "criterion"),
    [
        ({"static_score": 0.2}, "A1"),
        ({"cut_score": 0.9}, "A2"),
        ({"frame_count": 4}, "A4"),
        ({"signal_bit_depth": 7.0}, "A5"),
        ({"peak_code": 0.9999}, "A6"),
    ],
)
def test_each_failure_names_its_criterion_and_why(
    change: dict[str, object], criterion: str
) -> None:
    verdict = scr.screen_scene(_scene(**change), scr.Measurement.AMPLITUDE)
    assert not verdict.admissible
    failed = dict(verdict.failed)
    assert criterion in failed
    assert failed[criterion], "a rejection must say what the value was"


# ------------------------------------------------- unassessable is not a pass


def test_a_criterion_the_catalogue_cannot_answer_is_not_silently_passed() -> None:
    """The failure mode this state exists for.

    A catalogue that records the container's bit depth, or a format's legal-range limits, gives a
    constant for every scene. Fed to a threshold it satisfies it, and the scene then looks
    identical to one that was actually checked.
    """
    scene = _scene(signal_bit_depth=None, peak_code=None, floor_code=None)
    verdict = scr.screen_scene(scene, scr.Measurement.AMPLITUDE)

    assert not verdict.admissible
    assert not verdict.rejected
    assert verdict.needs_frame_check
    assert verdict.usable
    assert set(verdict.unassessed) == {"A5", "A6"}


def test_unassessed_does_not_apply_where_the_criterion_is_not_required() -> None:
    scene = _scene(signal_bit_depth=None, peak_code=None, floor_code=None)
    verdict = scr.screen_scene(scene, scr.Measurement.HETEROGENEITY)
    assert verdict.admissible
    assert not verdict.unassessed


def test_a_definite_failure_outranks_an_unassessable_one() -> None:
    scene = _scene(static_score=0.1, signal_bit_depth=None, peak_code=None, floor_code=None)
    verdict = scr.screen_scene(scene, scr.Measurement.AMPLITUDE)
    assert verdict.rejected
    assert not verdict.usable
    assert not verdict.needs_frame_check


# ------------------------------------------------------------------- reports


def test_the_report_separates_rejected_from_pending() -> None:
    scenes = [
        _scene(scene_id="ok"),
        _scene(scene_id="moving", static_score=0.1),
        _scene(scene_id="pending", signal_bit_depth=None, peak_code=None, floor_code=None),
    ]
    report = scr.screen(scenes, scr.Measurement.AMPLITUDE)
    assert [s.scene_id for s in report.admissible] == ["ok"]
    assert [s.scene_id for s in report.needs_frame_check] == ["pending"]
    assert [s.scene_id for s in report.usable] == ["ok", "pending"]


def test_the_report_names_what_the_catalogue_could_not_answer() -> None:
    scenes = [_scene(signal_bit_depth=None, peak_code=None, floor_code=None)]
    report = scr.screen(scenes, scr.Measurement.AMPLITUDE)
    assert report.unassessed_criteria()
    assert "UNASSESSED" in report.summary()


def test_changed_thresholds_are_recorded_and_printed() -> None:
    """Screening never widens a threshold itself; a caller who does cannot hide it."""
    loose = scr.ScreeningThresholds(min_static_score=0.1)
    report = scr.screen([_scene(static_score=0.2)], scr.Measurement.TEMPORAL, thresholds=loose)
    assert not report.thresholds_are_default
    assert "[THRESHOLDS CHANGED]" in report.summary()
    assert report.as_record()["thresholds"]["min_static_score"] == 0.1


def test_coverage_potential_uses_the_level_span_not_the_average() -> None:
    """A scene spanning shadow to highlight can supply windows in both.

    Judging it by its mean is how a catalogue of 304 scenes reads as almost entirely midtone.
    """
    wide = _scene(median_level=0.1, level_p10=0.001, level_p90=0.5)
    potential = scr.screen([wide], scr.Measurement.TEMPORAL).coverage_potential()
    assert potential["shadow"] == 1
    assert potential["highlight"] == 1


def test_an_empty_band_is_reported_as_a_property_of_the_corpus() -> None:
    dark = _scene(level_p10=0.001, level_p90=0.01)
    report = scr.screen([dark], scr.Measurement.TEMPORAL)
    assert report.coverage_potential()["highlight"] == 0
    assert "NO MATERIAL" in report.summary()


def test_screen_all_answers_every_measurement_separately() -> None:
    reports = scr.screen_all([_scene(resampled=True)])
    assert set(reports) == set(scr.Measurement)
    assert not reports[scr.Measurement.SPECTRUM].usable
    assert reports[scr.Measurement.TEMPORAL].usable


def test_the_report_serialises_with_its_criteria_and_thresholds() -> None:
    payload = scr.screen([_scene()], scr.Measurement.AMPLITUDE).as_record()
    for key in (
        "measurement",
        "required_criteria",
        "thresholds",
        "thresholds_are_default",
        "screened",
        "admissible",
        "needs_frame_check",
        "usable",
        "rejection_reasons",
        "unassessed_criteria",
        "coverage_potential",
    ):
        assert key in payload, key
