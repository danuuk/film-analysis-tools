"""The three admissibility checks the legacy catalogue could not make."""

from __future__ import annotations

import numpy as np
import pytest

from film_analysis_tools.capabilities.measure import admissibility as adm
from film_analysis_tools.capabilities.measure import synthetic, windows
from film_analysis_tools.core.errors import DataError

SIGMA = 0.01


def _frames(**kwargs: object) -> np.ndarray:
    base: dict[str, object] = {
        "frames": 16,
        "height": 256,
        "width": 256,
        "sigma": SIGMA,
        "seed": 9,
    }
    base.update(kwargs)
    return synthetic.sequence(synthetic.SyntheticSpec(**base))  # type: ignore[arg-type]


def _ramped(slope: float, **kwargs: object) -> np.ndarray:
    frames = _frames(**kwargs)
    return frames + (np.arange(frames.shape[0]) * slope)[:, None, None]


# ------------------------------------------------------------ A3: fades and ramps


def test_a_static_scene_is_not_ramping() -> None:
    evidence = adm.ramp_evidence(_frames())
    assert not evidence.is_ramping
    assert evidence.variance_share < 0.01


@pytest.mark.parametrize("slope", [0.005, 0.01, 0.02])
def test_a_fade_is_detected(slope: float) -> None:
    evidence = adm.ramp_evidence(_ramped(slope))
    assert evidence.is_ramping
    assert evidence.linearity > 0.95, "a steady fade should fit a straight line"


def test_the_variance_share_matches_the_closed_form() -> None:
    """The ramp contributes ``slope**2`` to a difference variance whose grain part is
    ``2 sigma**2``, so the share is ``slope**2 / (2 sigma**2)``. Being an identity rather than a
    heuristic is what lets one threshold serve every source."""
    slope = 0.005
    evidence = adm.ramp_evidence(_ramped(slope))
    expected = slope**2 / (2.0 * SIGMA**2)
    assert evidence.variance_share == pytest.approx(expected, rel=0.15)


def test_a_ramp_too_small_to_matter_is_not_flagged() -> None:
    """Rejecting a scene for a fade that contributes 0.5% of the variance would discard
    usable material for nothing."""
    assert not adm.ramp_evidence(_ramped(0.001)).is_ramping


def test_a3_catches_ramps_a_fixed_motion_gate_passes() -> None:
    """Why this is not redundant with the motion gate.

    A uniform ramp is low-frequency, so the gate does see it — but as an absolute quantity, with
    no knowledge of how damaging it is. The same 0.003-per-frame ramp is harmless against strong
    grain and *the entire measurement* against weak grain, and a fixed gate treats both alike.
    A share of the variance being measured scales with the grain; an RMS threshold cannot.
    """
    slope = 0.003
    strong = _ramped(slope, sigma=0.02)
    weak = _ramped(slope, sigma=0.002)

    assert windows.select_windows(weak, size=96).accepted, "the fixed gate passes this ramp"
    assert not adm.ramp_evidence(strong).is_ramping
    assert adm.ramp_evidence(weak).is_ramping
    assert adm.ramp_evidence(weak).variance_share > 0.5


def test_ramp_evidence_needs_enough_frames() -> None:
    with pytest.raises(DataError, match="at least 3 frames"):
        adm.ramp_evidence(_frames(frames=2))


# ---------------------------------------------------------------------- A6: clipping


def test_normal_material_is_not_clipped() -> None:
    evidence = adm.clipping_evidence(_frames(base_level=0.2))
    assert not evidence.is_clipped
    assert evidence.total_fraction < 0.001


def test_blown_material_is_clipped() -> None:
    evidence = adm.clipping_evidence(np.clip(_frames(base_level=1.05), 0.0, 1.0))
    assert evidence.is_clipped
    assert evidence.high_fraction > 0.5


def test_the_peak_alone_cannot_distinguish_these() -> None:
    """A specular highlight and a blown frame report the same peak code; only one destroys the
    distribution evidence. That is the gap this metric closes."""
    speck = _frames(base_level=0.2).copy()
    speck[:, :2, :2] = 1.0
    blown = np.clip(_frames(base_level=1.05), 0.0, 1.0)

    assert float(speck.max()) == pytest.approx(float(blown.max()), abs=1e-9)
    assert not adm.clipping_evidence(speck).is_clipped
    assert adm.clipping_evidence(blown).is_clipped


def test_floor_clipping_is_measured_too() -> None:
    crushed = np.clip(_frames(base_level=-0.02), 0.0, 1.0)
    evidence = adm.clipping_evidence(crushed)
    assert evidence.low_fraction > 0.5
    assert evidence.is_clipped


def test_limits_can_be_set_for_legal_range_material() -> None:
    frames = np.full((4, 64, 64), 940.0 / 1023.0)
    assert adm.clipping_evidence(frames, ceiling=940.0 / 1023.0, floor=64.0 / 1023.0).is_clipped


# ---------------------------------------------------------------------- A7: overlays


def _with_graphic(fraction_rows: float) -> np.ndarray:
    frames = _frames().copy()
    rows = int(frames.shape[1] * fraction_rows)
    if rows:
        frames[:, :rows, :] = 0.5  # a flat, perfectly noise-free band
    return frames


def test_clean_material_has_no_noise_free_regions() -> None:
    evidence = adm.overlay_evidence(_frames())
    assert evidence.noise_free_fraction == 0.0
    assert not evidence.has_overlay
    assert not evidence.any_overlay


def test_a_large_graphic_disqualifies_the_scene() -> None:
    evidence = adm.overlay_evidence(_with_graphic(0.25))
    assert evidence.has_overlay
    assert evidence.noise_free_fraction > 0.2


def test_a_small_graphic_is_masked_rather_than_disqualifying() -> None:
    """A corner logo should stop a window being placed on it, not condemn the whole scene."""
    frames = _frames().copy()
    frames[:, :32, :32] = 0.5  # one block: 1.6% of the frame
    evidence = adm.overlay_evidence(frames, block_size=32)

    assert not evidence.has_overlay, "one block is far below the disqualification threshold"
    assert evidence.any_overlay
    assert evidence.excludes(0, 0, 96)
    assert not evidence.excludes(128, 128, 64)


def test_a_graphic_thinner_than_one_block_is_not_detected() -> None:
    """A real limit of the method, recorded rather than assumed away.

    Detection is block-wise, so a graphic thinner than ``block_size`` is averaged away against
    the grain around it. That is why the block size is a parameter and not a constant.
    """
    assert not adm.overlay_evidence(_with_graphic(0.03), block_size=32).any_overlay
    assert adm.overlay_evidence(_with_graphic(0.03), block_size=4).any_overlay


def test_window_selection_refuses_windows_over_a_graphic() -> None:
    """The point of the mask: a noise-free region included in a window drags amplitude down,
    and it is perfectly static, so every staticness gate welcomes it."""
    frames = _with_graphic(0.20)
    overlay = adm.overlay_evidence(frames)
    without = windows.select_windows(frames, size=64)
    guarded = windows.select_windows(frames, size=64, overlay=overlay)

    assert len(guarded.accepted) < len(without.accepted)
    assert "overlaps a composited region" in guarded.rejection_reasons()
    for window in guarded.accepted:
        assert not overlay.excludes(window.x, window.y, window.size)


def test_overlay_evidence_needs_a_frame_pair_and_room_for_blocks() -> None:
    with pytest.raises(DataError, match="at least 2 frames"):
        adm.overlay_evidence(np.zeros((1, 128, 128)))
    with pytest.raises(DataError, match="too small"):
        adm.overlay_evidence(_frames(height=32, width=32), block_size=32)


# ------------------------------------------------------------------------ combined


def test_a_clean_scene_is_admissible() -> None:
    result = adm.scene_admissibility(_frames())
    assert result.admissible
    assert result.summary() == "admissible"
    assert not result.reasons


def test_every_failure_names_itself() -> None:
    ramping = adm.scene_admissibility(_ramped(0.02))
    assert not ramping.admissible
    assert any("ramp" in reason for reason in ramping.reasons)

    blown = adm.scene_admissibility(np.clip(_frames(base_level=1.05), 0.0, 1.0))
    assert any("clipped" in reason for reason in blown.reasons)

    titled = adm.scene_admissibility(_with_graphic(0.25))
    assert any("temporal noise" in reason for reason in titled.reasons)


def test_admissibility_serialises_with_all_three_checks() -> None:
    payload = adm.scene_admissibility(_frames()).as_record()
    assert set(payload) == {"admissible", "reasons", "ramp", "clipping", "overlay"}
    assert "variance_share" in payload["ramp"]
    assert "total_fraction" in payload["clipping"]
    assert "noise_free_fraction" in payload["overlay"]
