"""Native-pixel review layout and CLI contracts."""

from __future__ import annotations

import numpy as np

from film_analysis_tools.cli import build_parser
from film_analysis_tools.studies.negative_grain_native_crops import (
    DISPLAY_VARIANT_IDS,
    NATIVE_CROPS,
    NATIVE_VARIANT_IDS,
    native_triptych,
)


def test_native_triptych_preserves_every_source_pixel() -> None:
    baseline = np.full((7, 11, 3), 0.2, dtype=np.float64)
    output = baseline.copy()
    output[2, 4] += (0.01, -0.01, 0.005)

    triptych = native_triptych(output, baseline, limit=0.05)

    assert triptych.shape == (7, 33, 3)
    assert triptych.dtype == np.uint8
    assert np.any(triptych[:, 11:22] < 128)
    assert np.any(triptych[:, 11:22] > 128)
    assert np.any(triptych[:, 22:] > 0)


def test_native_review_is_exactly_the_three_bounded_n2_strengths() -> None:
    assert NATIVE_VARIANT_IDS == ("n0", "n2_075", "n2_100", "n2_125")
    assert DISPLAY_VARIANT_IDS == ("n2_075", "n2_100", "n2_125")
    assert {crop.crop_id for crop in NATIVE_CROPS} == {
        "skin_proxy",
        "deep_shadow",
        "neutral_midtone",
        "moving_colour",
    }


def test_native_crop_cli_defaults_to_four_seconds_and_four_frame_workers() -> None:
    args = build_parser().parse_args(
        [
            "negative-grain-native-crops",
            "--n1-bundle",
            "n1",
            "--n2-bundle",
            "n2",
            "--report",
            "report",
        ]
    )

    assert args.frames == 96
    assert args.frame_workers == 4
    assert args.delta_limit == 0.05
