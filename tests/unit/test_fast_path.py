"""The fast path, tested against transforms whose effect is known in closed form.

The centrepiece is ``test_identity_against_itself_shows_no_effect``: a harness that reports an
effect where there is none is not evidence about anything else. Everything downstream depends
on that check, so it is the first thing asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from film_analysis_tools.capabilities.colour import features, metrics, transforms
from film_analysis_tools.capabilities.sample import cohorts
from film_analysis_tools.capabilities.sample.table import RGB_COLUMN, SampleTable
from film_analysis_tools.capabilities.statistics import compare, compare_cohorts, shuffled_labels
from film_analysis_tools.core import SelectionError, Tier


@pytest.fixture
def table() -> SampleTable:
    """A deterministic spread of colours across hue, saturation and luma."""
    generator = np.random.default_rng(20260728)
    rgb = generator.uniform(0.01, 1.0, size=(4000, 3))
    return SampleTable(columns={RGB_COLUMN: rgb}, name="synthetic")


# ------------------------------------------------------------------ colour features


def test_hue_of_primaries_is_where_it_should_be() -> None:
    primaries = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float64
    )
    assert np.allclose(features.hue_degrees(primaries), [0.0, 120.0, 240.0, 60.0])


def test_greys_are_unsaturated_and_hue_free() -> None:
    greys = np.asarray([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]], dtype=np.float64)
    assert np.allclose(features.saturation(greys), 0.0)
    names = features.hue_sector_names(features.hue_degrees(greys), features.saturation(greys))
    assert list(names) == ["neutral", "neutral"]


def test_hue_sectors_name_the_expected_regions() -> None:
    samples = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    names = features.hue_sector_names(features.hue_degrees(samples), features.saturation(samples))
    assert list(names) == ["red", "green", "blue"]


# -------------------------------------------------------------------- sample table


def test_where_supports_every_documented_operator(table: SampleTable) -> None:
    assert len(table.where(luma_bt2020__gt=0.5)) < len(table)
    assert len(table.where(luma_bt2020__lt=0.5)) < len(table)
    assert len(table.where(luma_bt2020__between=(0.2, 0.8))) < len(table)
    assert len(table.where(hue_sector__in=("red", "green"))) < len(table)
    assert len(table.where(hue_sector="red")) < len(table)
    assert len(table.where(hue_sector__ne="red")) < len(table)


def test_where_composes_and_records_the_cohort_expression(table: SampleTable) -> None:
    selected = table.where(luma_bt2020__gt=0.3, relative_saturation__gt=0.2)
    assert len(selected) <= len(table.where(luma_bt2020__gt=0.3))
    assert "luma_bt2020 > 0.3" in selected.label
    assert "relative_saturation > 0.2" in selected.label


def test_derived_columns_work_on_a_pack_that_never_stored_them(table: SampleTable) -> None:
    assert table.column_names == [RGB_COLUMN]
    assert table.column("hue_sector").shape[0] == len(table)
    assert table.column("luma_bt2020").shape[0] == len(table)


def test_unknown_column_names_the_alternatives(table: SampleTable) -> None:
    with pytest.raises(SelectionError, match="unknown column"):
        table.column("nonexistent")


def test_selection_preserves_row_alignment_across_columns() -> None:
    rgb = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    tagged = SampleTable(columns={RGB_COLUMN: rgb, "tag": np.asarray(["a", "b", "c"])})
    green = tagged.where(hue_sector="green")
    assert list(green.column("tag")) == ["b"]


# ----------------------------------------------------------------------- transforms


def test_identity_changes_nothing(table: SampleTable) -> None:
    assert np.array_equal(transforms.identity()(table.rgb), table.rgb)


def test_exposure_moves_luma_but_not_hue(table: SampleTable) -> None:
    brighter = transforms.exposure(1.0)(table.rgb)
    assert np.allclose(metrics.luma_ratio(table.rgb, brighter), 1.0)
    assert np.allclose(metrics.hue_drift(table.rgb, brighter), 0.0, atol=1e-9)


def test_tone_gamma_preserves_hue(table: SampleTable) -> None:
    curved = transforms.tone_gamma(0.8)(table.rgb)
    assert np.allclose(metrics.hue_drift(table.rgb, curved), 0.0, atol=1e-9)


def test_saturate_increases_chroma_without_moving_luma(table: SampleTable) -> None:
    saturated = transforms.saturate(1.2)(table.rgb)
    assert float(np.median(metrics.chroma_delta(table.rgb, saturated))) > 0.0
    assert np.allclose(metrics.luma_ratio(table.rgb, saturated), 0.0, atol=1e-9)


def test_hue_drift_wraps_the_short_way_around() -> None:
    before = np.asarray([[1.0, 0.02, 0.0]], dtype=np.float64)  # just above 0 deg
    after = np.asarray([[1.0, 0.0, 0.02]], dtype=np.float64)  # just below 360 deg
    drift = metrics.hue_drift(before, after)
    assert abs(float(drift[0])) < 10.0


# ------------------------------------------------------------------- null controls


def test_null_control_on_a_zero_effect_lands_near_zero() -> None:
    generator = np.random.default_rng(1)
    noise = generator.normal(0.0, 1.0, size=2000)
    result = shuffled_labels(noise, float(np.median(noise)), resamples=200, seed=7)
    assert result.is_clean
    assert result.p_value > 0.05


def test_null_control_rejects_a_large_real_effect() -> None:
    shifted = np.full(2000, 5.0) + np.random.default_rng(2).normal(0.0, 0.1, size=2000)
    result = shuffled_labels(shifted, float(np.median(shifted)), resamples=200, seed=7)
    assert result.p_value < 0.05


def test_null_p_value_is_bounded_away_from_an_unearned_zero() -> None:
    values = np.full(500, 100.0)
    result = shuffled_labels(values, 100.0, resamples=100, seed=3)
    assert result.p_value >= 1.0 / 101.0


# ------------------------------------------------------------- the known-answer test


def test_identity_against_itself_shows_no_effect(table: SampleTable) -> None:
    """The harness's own null control. If this fails, nothing else it reports means anything."""
    result = compare(
        table,
        baseline=transforms.identity(),
        candidate=transforms.identity(),
        metric="hue_drift",
    )
    assert result.effect == 0.0
    assert result.spread == 0.0
    assert not result.exceeds_null
    assert result.count == len(table)


def test_a_known_effect_is_detected_when_the_cohort_shares_a_direction(
    table: SampleTable,
) -> None:
    """A red gain pulls hue toward red, so within one hue sector the drift has one sign."""
    cohort = table.where(hue_sector="green", relative_saturation__gt=0.2)
    result = compare(
        cohort,
        baseline=transforms.identity(),
        candidate=transforms.channel_gain(red=1.15),
        metric="hue_drift",
    )
    assert result.is_directional
    assert result.exceeds_null
    assert result.null.p_value < 0.05
    assert result.unit == "deg"


def test_a_cancelling_effect_is_reported_as_such_and_not_as_no_effect(
    table: SampleTable,
) -> None:
    """The subtlety that a bare verdict would hide.

    Over a full hue circle a red gain moves every sample, but samples on opposite sides move
    opposite ways, so the median signed drift is ~0 and the null cannot reject it. Reporting
    "no effect" there would be wrong: the magnitude is large and the cohort is simply too
    broad for the question.
    """
    result = compare(
        table,
        baseline=transforms.identity(),
        candidate=transforms.channel_gain(red=1.15),
        metric="hue_drift",
    )
    assert abs(result.effect) < 0.5
    assert result.magnitude > 1.0
    assert not result.is_directional
    assert result.verdict == "moves, no net direction"


def test_effect_direction_follows_the_transform(table: SampleTable) -> None:
    warmer = compare(
        table,
        baseline=transforms.identity(),
        candidate=transforms.saturate(1.3),
        metric="chroma_delta",
    )
    duller = compare(
        table,
        baseline=transforms.identity(),
        candidate=transforms.saturate(0.7),
        metric="chroma_delta",
    )
    assert warmer.effect > 0.0 > duller.effect


def test_comparison_reports_tier_and_never_a_bare_verdict(table: SampleTable) -> None:
    result = compare(table, baseline=transforms.identity(), candidate=transforms.saturate(1.2))
    text = result.summary()
    assert result.tier is Tier.COMPARISON
    for expected in ("spread", "n=", "null", "comparison"):
        assert expected in text


# ------------------------------------------------------------------------- cohorts


def test_cohorts_partition_by_colour_and_stay_non_empty(table: SampleTable) -> None:
    built = cohorts.build(table, ("neutral", "foliage_like", "shadows", "highlights"))
    assert built
    for name, cohort in built.items():
        assert len(cohort) > 0, name
        assert len(cohort) < len(table), name


def test_unknown_cohort_names_the_alternatives(table: SampleTable) -> None:
    with pytest.raises(SelectionError, match="unknown cohorts"):
        cohorts.build(table, ("not_a_cohort",))


def test_compare_across_cohorts_labels_each_result(table: SampleTable) -> None:
    built = cohorts.build(table, ("neutral", "foliage_like"))
    results = compare_cohorts(
        built,
        baseline=transforms.identity(),
        candidate=transforms.channel_gain(green=1.1),
        metric="hue_drift",
    )
    assert [result.cohort for result in results] == list(built)
