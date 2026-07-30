"""Window selection: the gate never relaxes, and thin coverage is loud."""

from __future__ import annotations

import numpy as np
import pytest

from film_analysis_tools.capabilities.measure import windows
from film_analysis_tools.core.errors import DataError


def _banded(frames: int = 8, size: int = 384, *, seed: int = 4) -> np.ndarray:
    """Three luma bands side by side, with texture on the top half."""
    generator = np.random.default_rng(seed)
    base = np.zeros((size, size))
    base[:, : size // 3] = 0.005
    base[:, size // 3 : 2 * size // 3] = 0.10
    base[:, 2 * size // 3 :] = 0.40
    base[: size // 2] += 0.02 * np.sin(np.arange(size) / 4.0)[None, :]
    return np.stack([base + generator.normal(0, 0.004, size=(size, size)) for _ in range(frames)])


def _uniformly_dark(frames: int = 8, size: int = 256) -> np.ndarray:
    generator = np.random.default_rng(11)
    return np.stack([0.004 + generator.normal(0, 0.004, size=(size, size)) for _ in range(frames)])


# ------------------------------------------------------------------------- selection


def test_windows_are_classified_across_level_texture_and_position() -> None:
    report = windows.select_windows(_banded(), size=96)
    assert report.accepted
    assert {window.band for window in report.accepted} <= set(windows.BANDS)
    assert {window.texture for window in report.accepted} <= set(windows.TEXTURES)
    assert {window.position for window in report.accepted} <= set(windows.POSITIONS)


def test_moving_regions_are_rejected_with_a_reason() -> None:
    frames = _banded()
    moving = frames.copy()
    for index in range(moving.shape[0]):
        moving[index, 288:, 288:] += 0.05 * np.sin((np.arange(96) + index * 9) / 5.0)[None, :]
    report = windows.select_windows(moving, size=96)
    assert report.rejected
    assert set(report.rejection_reasons()) <= {"motion above gate", "sub-pixel drift"}


def test_coverage_is_reported_per_band() -> None:
    report = windows.select_windows(_banded(), size=96)
    coverage = report.coverage()
    assert set(coverage) == set(windows.BANDS)
    assert sum(coverage.values()) == len(report.accepted)


def test_the_measured_level_range_is_recorded() -> None:
    """The legacy curve spanned luma 0.00012 to 0.282 and nothing said so at selection time."""
    low, high = windows.select_windows(_banded(), size=96).measured_level_range()
    assert low < 0.02 < high


# ------------------------------------------------------ reject rather than relax


def test_material_that_cannot_support_a_curve_says_so_before_fitting() -> None:
    report = windows.select_windows(_uniformly_dark(), size=96)
    assert not report.sufficient
    assert "INSUFFICIENT" in report.summary()


def test_partial_coverage_is_distinguished_from_insufficiency() -> None:
    """Two different statements. Printing one while the flag says the other is how a thin
    result gets mistaken for a complete one."""
    report = windows.select_windows(_banded(), size=96)
    summary = report.summary()
    if report.sufficient and report.missing_bands():
        assert "PARTIAL COVERAGE" in summary
        assert "INSUFFICIENT" not in summary
    if not report.sufficient:
        assert "INSUFFICIENT" in summary


def test_a_widened_gate_is_recorded_in_the_report() -> None:
    """The gate is never relaxed automatically, and a caller who relaxes it cannot hide it."""
    wide = windows.WindowGate(max_motion_energy=0.05)
    report = windows.select_windows(_banded(), size=96, gate=wide)
    assert not report.gate_is_default
    assert report.as_record()["gate"]["max_motion_energy"] == 0.05
    assert "[GATE WIDENED]" in report.summary()


def test_the_default_gate_is_reported_as_default() -> None:
    report = windows.select_windows(_banded(), size=96)
    assert report.gate_is_default
    assert "[GATE WIDENED]" not in report.summary()


def test_selection_never_widens_the_gate_itself() -> None:
    """Yield never buys a looser threshold: nothing passing means nothing passes."""
    strict = windows.WindowGate(max_motion_energy=1e-9, max_subpixel_residual=1e-9)
    report = windows.select_windows(_banded(), size=96, gate=strict)
    assert not report.accepted
    assert report.gate == strict
    assert not report.sufficient
    assert any("not a grain-amplitude source" in note for note in report.notes)


# ------------------------------------------------------------------------ stratified


def test_stratified_selection_spreads_across_strata() -> None:
    """Taking the best-scoring windows is how a corpus ends up measured only in the shadows:
    the quietest regions of most footage are the dark ones."""
    report = windows.select_windows(_banded(), size=96)
    chosen = windows.stratified(report, per_stratum=1)
    strata = [window.stratum for window in chosen]
    assert len(strata) == len(set(strata))
    assert len({stratum[0] for stratum in strata}) >= 2


# ------------------------------------------------------------------------- contracts


def test_too_few_frames_is_refused() -> None:
    with pytest.raises(DataError, match="at least"):
        windows.select_windows(_banded(frames=3), size=96)


def test_an_oversized_window_is_refused() -> None:
    with pytest.raises(DataError, match="exceeds the frame"):
        windows.select_windows(_banded(size=128), size=256)


def test_report_serialises_with_its_gate_and_coverage() -> None:
    payload = windows.select_windows(_banded(), size=96).as_record()
    for key in (
        "gate",
        "gate_is_default",
        "coverage",
        "missing_bands",
        "measured_level_min",
        "measured_level_max",
        "sufficient",
        "rejection_reasons",
    ):
        assert key in payload, key


def test_a_window_slices_the_frames_it_describes() -> None:
    frames = _banded()
    window = windows.select_windows(frames, size=96).accepted[0]
    region = window.slice_of(frames)
    assert region.shape == (frames.shape[0], 96, 96)
