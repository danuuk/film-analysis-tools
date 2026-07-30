"""The region record: what a measured tile must carry to be worth keeping."""

from __future__ import annotations

import numpy as np
import pytest

from film_analysis_tools.capabilities.catalogue import regions as rg
from film_analysis_tools.capabilities.measure import synthetic
from film_analysis_tools.capabilities.measure import windows as win
from film_analysis_tools.core.errors import SelectionError


def _window(**kwargs: object) -> win.Window:
    base: dict[str, object] = {
        "x": 0,
        "y": 0,
        "size": 128,
        "level": 0.1,
        "motion_energy": 0.001,
        "structure_snr": 0.5,
        "subpixel_residual": 0.05,
        "band": "midtone",
        "texture": "flat",
        "position": "centre",
    }
    base.update(kwargs)
    return win.Window(**base)  # type: ignore[arg-type]


def _provenance(**kwargs: object) -> rg.RegionProvenance:
    base: dict[str, object] = {
        "source_id": "film",
        "interval_start_s": 10.0,
        "interval_end_s": 12.0,
        "start_frame": 240,
        "frames": 48,
        "level_scale": "linear",
    }
    base.update(kwargs)
    return rg.RegionProvenance(**base)  # type: ignore[arg-type]


def _region(*, window: object = None, **provenance: object) -> rg.Region:
    return rg.Region(
        window=window or _window(),  # type: ignore[arg-type]
        provenance=_provenance(**provenance),
    )


# ------------------------------------------------------------------ the record


def test_a_region_is_addressable() -> None:
    """Gap 5: a fitted knot must be able to name the regions behind it."""
    region = _region(window=_window(x=256, y=128))
    assert region.region_id == "film@240+48:256,128+128"


def test_a_level_without_a_scale_is_refused() -> None:
    """0.18 linear and 0.18 in PQ code are different amounts of light. Pooling them silently
    produces a curve that is wrong in a way nothing downstream would notice."""
    with pytest.raises(SelectionError, match="unknown level scale"):
        _provenance(level_scale="whatever")


def test_an_inverted_interval_is_refused() -> None:
    with pytest.raises(SelectionError, match="ends before it starts"):
        _provenance(interval_start_s=12.0, interval_end_s=10.0)


def test_a_region_spanning_no_frames_is_refused() -> None:
    with pytest.raises(SelectionError, match="at least one frame"):
        _provenance(frames=0)


def test_colour_and_face_are_absent_not_zero() -> None:
    """A tile from a luma stack has no colour to report; 0.0 saturation would be a claim."""
    region = _region()
    assert region.colour is None
    assert region.face is None
    assert region.as_record()["colour"] is None


def test_the_window_measurements_are_reachable_without_unwrapping() -> None:
    region = _region(window=_window(level=0.3, band="highlight", texture="textured"))
    assert (region.level, region.band, region.texture) == (0.3, "highlight", "textured")
    assert region.stratum == ("highlight", "textured", "centre")


# ---------------------------------------------------- capture from a real report


def test_a_selection_report_becomes_indexed_regions() -> None:
    """The step that stops throwing the work away."""
    frames = synthetic.sequence(synthetic.SyntheticSpec(frames=6, sigma=0.02, rho=0.0))
    report = win.select_windows(frames, size=64)
    assert report.accepted

    built = rg.regions_from_report(
        report,
        source_id="film",
        interval_start_s=4.0,
        interval_end_s=6.0,
        start_frame=96,
        level_scale="linear",
    )
    assert len(built) == len(report.accepted)
    assert all(r.provenance.frames == report.frames for r in built)
    assert all(r.provenance.gate_is_default for r in built)
    assert len({r.region_id for r in built}) == len(built)


def test_only_accepted_windows_become_regions() -> None:
    """A rejection describes why a run failed; it is not a catalogue entry."""
    moving = synthetic.sequence(
        synthetic.SyntheticSpec(
            frames=6, sigma=0.01, rho=0.0, texture_amplitude=0.3, drift_px_per_frame=(0.0, 2.0)
        )
    )
    report = win.select_windows(moving, size=64)
    built = rg.regions_from_report(
        report,
        source_id="film",
        interval_start_s=0.0,
        interval_end_s=2.0,
        start_frame=0,
        level_scale="linear",
    )
    assert report.rejected
    assert len(built) == len(report.accepted) < len(report.accepted) + len(report.rejected)


# ----------------------------------------------------------------- independence


def test_regions_tiled_from_one_interval_are_not_many_samples() -> None:
    """The count that matters. 40 tiles of the same two seconds is one span, not 40."""
    idx = rg.index([_region(window=_window(x=128 * i)) for i in range(40)])
    report = idx.independence()
    assert report.regions == 40
    assert report.intervals == 1
    assert report.spans == 1
    assert report.concentrated
    assert "treat ~1 as the independent count, not 40" in report.summary()
    assert "CONCENTRATED" in report.summary()


def test_overlapping_intervals_do_not_count_twice() -> None:
    """Step 1 builds 2 s intervals on a 1 s stride, so neighbours share half their frames."""
    built = [
        _region(interval_start_s=float(i), interval_end_s=float(i) + 2.0, start_frame=i * 24)
        for i in range(5)
    ]
    report = rg.index(built).independence()
    assert report.intervals == 5
    assert report.spans == 1  # 0..2, 1..3, ... 4..6 all merge
    assert report.span_seconds == pytest.approx(6.0)


def test_disjoint_intervals_are_counted_separately() -> None:
    built = [
        _region(interval_start_s=0.0, interval_end_s=2.0),
        _region(interval_start_s=100.0, interval_end_s=102.0),
    ]
    report = rg.index(built).independence()
    assert report.spans == 2
    assert report.span_seconds == pytest.approx(4.0)


def test_a_spread_corpus_is_not_flagged_as_concentrated() -> None:
    built = [
        _region(interval_start_s=float(10 * i), interval_end_s=float(10 * i) + 2.0)
        for i in range(10)
    ]
    assert not rg.index(built).independence().concentrated


def test_empty_regions_do_not_divide_by_zero() -> None:
    report = rg.index([]).independence()
    assert (report.spans, report.regions_per_span, report.concentrated) == (0, 0.0, False)


# ---------------------------------------------------------------------- queries


def test_selection_filters_on_every_stated_condition() -> None:
    idx = rg.index(
        [
            _region(window=_window(band="shadow", level=0.01)),
            _region(window=_window(band="midtone", level=0.10, texture="textured")),
            _region(window=_window(band="highlight", level=0.40, position="edge")),
        ]
    )
    assert len(idx.select(band="shadow")) == 1
    assert len(idx.select(texture="textured")) == 1
    assert len(idx.select(position="edge")) == 1
    assert len(idx.select(min_level=0.05)) == 2
    assert len(idx.select(max_level=0.05)) == 1


def test_a_widened_gate_must_be_asked_for() -> None:
    """Yield is never a reason to relax a gate, and the relaxation must stay visible."""
    wide = win.WindowGate(max_motion_energy=0.5)
    idx = rg.index([_region(), _region(gate=wide)])
    assert len(idx.select()) == 1
    assert len(idx.select(require_default_gate=False)) == 2
    assert idx.widened_gate_count() == 1
    assert "widened gate" in idx.summary()


def test_a_per_interval_cap_forces_spread_over_depth() -> None:
    built = [
        _region(interval_start_s=float(10 * i), interval_end_s=float(10 * i) + 2.0)
        for i in range(4)
        for _ in range(25)
    ]
    idx = rg.index(built)
    assert len(idx) == 100
    capped = idx.select(per_interval_cap=5)
    assert len(capped) == 20
    assert rg.index(capped).independence().spans == 4


def test_colour_and_face_predicates_need_the_evidence_present() -> None:
    plain = _region()
    coloured = rg.Region(
        window=_window(), provenance=_provenance(), colour=rg.RegionColour(30.0, 15.0)
    )
    facial = rg.Region(
        window=_window(), provenance=_provenance(), face=rg.RegionFace(mesh_fraction=0.8)
    )
    idx = rg.index([plain, coloured, facial])
    assert idx.select(require_colour=True) == [coloured]
    assert idx.select(require_face=True) == [facial]
    assert idx.select(min_face_fraction=0.9) == []


def test_an_unknown_predicate_value_is_refused() -> None:
    idx = rg.index([_region()])
    with pytest.raises(SelectionError, match="unknown band"):
        idx.select(band="chartreuse")
    with pytest.raises(SelectionError, match="unknown level scale"):
        idx.select(level_scale="furlongs")


# --------------------------------------------------------------------- reporting


def test_mixed_level_scales_are_called_out() -> None:
    idx = rg.index([_region(level_scale="linear"), _region(level_scale="pq_code")])
    assert idx.level_scales() == ["linear", "pq_code"]
    assert "MIXED SCALES" in idx.summary()
    assert "must not be pooled" in idx.summary()


def test_a_single_scale_is_not_flagged() -> None:
    assert "MIXED SCALES" not in rg.index([_region(), _region()]).summary()


def test_band_edges_above_all_the_data_are_called_out() -> None:
    """Measured on the real 4K PQ master: linear levels spanned 0.00000-0.02865 while the
    default edges are (0.02, 0.25), so 737 of 739 regions were labelled shadow. That reads as
    "no highlights in this film" and is really "diffuse white sits near 0.01 on this scale".
    """
    idx = rg.index([_region(window=_window(level=level)) for level in (0.0001, 0.002, 0.008)])
    assert not idx.edges_bracket_the_data
    assert "EDGES DO NOT BRACKET THE DATA" in idx.summary()
    assert "units mismatch" in idx.summary()


def test_edges_inside_the_measured_range_are_not_flagged() -> None:
    idx = rg.index([_region(window=_window(level=level)) for level in (0.001, 0.1, 0.4)])
    assert idx.edges_bracket_the_data
    assert "EDGES DO NOT BRACKET" not in idx.summary()


def test_an_empty_index_makes_no_claim_about_edges() -> None:
    assert rg.index([]).edges_bracket_the_data


def test_the_index_serialises() -> None:
    record = rg.index([_region()]).as_record()
    for key in ("regions", "coverage", "level_scales", "widened_gate", "independence"):
        assert key in record, key
    assert record["independence"]["spans"] == 1


def test_a_region_serialises_with_its_provenance() -> None:
    record = _region().as_record()
    assert record["region_id"] and record["provenance"]["level_scale"] == "linear"
    assert record["provenance"]["gate_is_default"] is True
    assert record["x"] == 0 and record["band"] == "midtone"


def test_merged_spans_are_reported_directly() -> None:
    spans, seconds = rg.merged_span_seconds(
        [
            _region(interval_start_s=0.0, interval_end_s=2.0),
            _region(interval_start_s=1.0, interval_end_s=3.0),
            _region(interval_start_s=50.0, interval_end_s=52.0),
        ]
    )
    assert (spans, seconds) == (2, pytest.approx(5.0))


def test_a_tile_mean_hides_what_the_tile_contains() -> None:
    """The same "judge by the average" error step 1 found at interval level, one level down.

    Measured on the 4K master: of 120 tiles in one interval, 18 reached the highlight edge by
    mean and 32 by content.
    """
    dark_with_lamp = _region(
        window=_window(level=0.01, band="shadow", level_low=0.005, level_high=0.40)
    )
    uniformly_dark = _region(
        window=_window(level=0.01, band="shadow", level_low=0.005, level_high=0.015)
    )
    assert dark_with_lamp.contains_band("highlight")
    assert not uniformly_dark.contains_band("highlight")

    idx = rg.index([dark_with_lamp, uniformly_dark])
    assert idx.coverage()["highlight"] == 0
    assert idx.coverage(contains=True)["highlight"] == 1
    assert idx.select(contains_band="highlight") == [dark_with_lamp]
    assert "(contains)" in idx.summary()


def test_contains_band_uses_the_edges_the_region_was_measured_with() -> None:
    region = _region(window=_window(level_low=0.005, level_high=0.40), band_edges=(0.02, 0.9))
    assert not region.contains_band("highlight")


def test_an_unknown_contains_band_is_refused() -> None:
    with pytest.raises(SelectionError, match="unknown band"):
        rg.index([_region()]).select(contains_band="chartreuse")


def test_levels_and_bands_are_summarised() -> None:
    idx = rg.index(
        [_region(window=_window(level=0.01, band="shadow")), _region(window=_window(level=0.4))]
    )
    assert idx.level_range() == (0.01, 0.4)
    assert idx.coverage()["shadow"] == 1
    assert np.isclose(idx.level_range()[1], 0.4)
