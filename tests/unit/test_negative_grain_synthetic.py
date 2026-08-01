"""Original-float metrics and fixed-scale visualization contracts."""

from __future__ import annotations

import numpy as np
import pytest

from film_analysis_tools.cli import build_parser
from film_analysis_tools.core import Tier
from film_analysis_tools.forward.negative_grain_synthetic import NegativeGrainRegion
from film_analysis_tools.studies.negative_grain_synthetic import (
    STUDY,
    RegionAccumulator,
    magnitude_delta_view,
    review_triptych,
    signed_delta_view,
)


def _region() -> NegativeGrainRegion:
    return NegativeGrainRegion(
        region_id="probe",
        category="neutral_step",
        description="controlled probe",
        motion="static",
        bounds_yxyx=(0, 8, 0, 8),
        scene_luma=0.18,
        exposure_stops_from_grey=0.0,
    )


def test_signed_delta_uses_neutral_grey_and_never_frame_normalizes() -> None:
    delta = np.asarray([[[-0.05, 0.0, 0.05]]], dtype=np.float64)

    signed = signed_delta_view(delta, limit=0.05)
    magnitude = magnitude_delta_view(delta, limit=0.05)

    np.testing.assert_array_equal(signed, [[[0.0, 0.5, 1.0]]])
    np.testing.assert_array_equal(magnitude, [[[1.0, 0.0, 1.0]]])
    weaker = signed_delta_view(delta * 0.5, limit=0.05)
    np.testing.assert_array_equal(weaker, [[[0.25, 0.5, 0.75]]])


def test_region_metrics_use_signed_float_delta_before_display_encoding() -> None:
    region = _region()
    accumulator = RegionAccumulator(
        region_id=region.region_id,
        category=region.category,
        description=region.description,
        motion=region.motion,
    )
    yy, xx = np.indices((8, 8))
    field = np.where((yy + xx) % 2 == 0, 1.0, -1.0)
    for frame_index in range(8):
        sign = 1.0 if frame_index % 2 == 0 else -1.0
        delta = np.stack((field, -0.5 * field, 0.25 * field), axis=2) * sign * 0.01
        accumulator.update(delta, 0.5 + delta, region, frame_index=frame_index)

    record = accumulator.as_record()

    assert record["temporal_luma_rms"] > 0.0
    assert record["opponent_rms"] > record["temporal_luma_rms"]
    assert record["mean_luma_delta"] == pytest.approx(0.0, abs=1.0e-15)
    np.testing.assert_allclose(record["mean_rgb_delta"], 0.0, atol=1.0e-15)
    assert record["p95_delta_magnitude"] > 0.0
    assert record["adjacent_frame_correlation"]["median"] == pytest.approx(-1.0)
    assert record["spatial"]["frames"] == 2
    assert len(record["spatial"]["radial_psd_normalized"]) == 24


def test_review_triptych_keeps_normal_signed_and_magnitude_views_separate() -> None:
    baseline = np.full((8, 8, 3), 0.2, dtype=np.float64)
    output = baseline.copy()
    output[::2, ::2] += 0.01
    output[1::2, 1::2] -= 0.01

    review = review_triptych(output, baseline, limit=0.05)

    assert review.shape == (8, 24, 3)
    assert review.dtype == np.uint8
    assert np.any(review[:, 8:16] < 128)
    assert np.any(review[:, 8:16] > 128)
    assert np.all(review[:, 16:] >= 0)


def test_study_is_a_comparison_with_a_real_null_and_cli_is_runnable() -> None:
    assert STUDY.tier is Tier.COMPARISON
    assert "null" in STUDY.controls
    parser = build_parser()
    args = parser.parse_args(
        [
            "negative-grain-synthetic",
            "--n1-bundle",
            "n1",
            "--n2-bundle",
            "n2",
            "--report",
            "report",
            "--no-video",
        ]
    )
    assert args.frames == 96
    assert args.width == 1920 and args.height == 1080
    assert args.frame_workers == 4
    assert args.variant_workers == 1
