"""The five evidence types, and the colour limitation enforced by type."""

from __future__ import annotations

import numpy as np
import pytest

from film_analysis_tools.capabilities.measure import chroma, evidence, synthetic, windows
from film_analysis_tools.core.errors import DataError, SelectionError

TRUE_SIGMA = 0.01


def _material(**kwargs: object) -> tuple[np.ndarray, tuple[windows.Window, ...]]:
    base: dict[str, object] = {
        "frames": 16,
        "height": 256,
        "width": 256,
        "sigma": TRUE_SIGMA,
        "seed": 9,
    }
    base.update(kwargs)
    frames = synthetic.sequence(synthetic.SyntheticSpec(**base))  # type: ignore[arg-type]
    return frames, windows.select_windows(frames, size=96).accepted


# ------------------------------------------------------------------- independence


def test_each_producer_runs_without_the_others() -> None:
    """Independence is the point: any one can be re-run or disbelieved on its own."""
    frames, selected = _material()
    assert evidence.amplitude_evidence(frames, selected).points
    assert evidence.spectrum_evidence(frames, selected).frequencies
    assert evidence.distribution_evidence(frames, selected).std > 0
    assert evidence.heterogeneity_evidence(frames, selected).sample_count > 0
    assert evidence.temporal_evidence(frames, selected).sample_count > 0


def test_every_producer_serialises_under_its_own_name() -> None:
    frames, selected = _material()
    names = {
        evidence.amplitude_evidence(frames, selected).as_record()["evidence"],
        evidence.spectrum_evidence(frames, selected).as_record()["evidence"],
        evidence.distribution_evidence(frames, selected).as_record()["evidence"],
        evidence.heterogeneity_evidence(frames, selected).as_record()["evidence"],
        evidence.temporal_evidence(frames, selected).as_record()["evidence"],
    }
    assert names == {
        "amplitude_vs_level",
        "spectrum",
        "distribution",
        "slow_heterogeneity",
        "temporal",
    }


def test_producers_refuse_an_empty_window_set() -> None:
    frames, _ = _material()
    for producer in (
        evidence.spectrum_evidence,
        evidence.distribution_evidence,
        evidence.heterogeneity_evidence,
        evidence.temporal_evidence,
    ):
        with pytest.raises(DataError):
            producer(frames, [])


# ------------------------------------------------------------------ 1. amplitude


def test_amplitude_recovers_the_known_sigma() -> None:
    frames, selected = _material()
    points = evidence.amplitude_evidence(frames, selected).points
    assert float(np.median([point.sigma for point in points])) == pytest.approx(
        TRUE_SIGMA, rel=0.05
    )


def test_amplitude_points_carry_their_own_trust() -> None:
    frames, selected = _material()
    result = evidence.amplitude_evidence(frames, selected)
    assert result.trusted, "clean synthetic grain should produce trusted points"
    assert all(isinstance(point.trustworthy, bool) for point in result.points)


# ------------------------------------------------------------------- 2. spectrum


def test_white_grain_measures_as_white() -> None:
    frames, selected = _material()
    result = evidence.spectrum_evidence(frames, selected)
    assert result.is_white
    assert abs(result.autocorrelation[0]) < 0.1


def test_spatially_correlated_grain_is_not_white() -> None:
    """Structure is what distinguishes film grain from sensor noise; it must be visible here."""
    frames, selected = _material(spatial_correlation_px=2.0)
    if not selected:
        pytest.skip("no windows accepted on this material")
    result = evidence.spectrum_evidence(frames, selected)
    assert not result.is_white
    assert result.whiteness < 0.8
    assert result.autocorrelation[0] > 0.2


# --------------------------------------------------------------- 3. distribution


def test_gaussian_grain_measures_as_gaussian() -> None:
    frames, selected = _material()
    result = evidence.distribution_evidence(frames, selected)
    assert result.is_gaussian
    assert result.std == pytest.approx(TRUE_SIGMA, rel=0.05)
    assert abs(result.excess_kurtosis) < 0.3


def test_a_heavy_tail_is_reported_rather_than_absorbed_into_sigma() -> None:
    """A single sigma only describes the residual when the distribution is near-Gaussian."""
    frames, selected = _material()
    spiked = frames.copy()
    generator = np.random.default_rng(2)
    mask = generator.random(spiked.shape) < 0.002
    spiked[mask] += 0.25
    result = evidence.distribution_evidence(spiked, selected)
    assert not result.is_gaussian
    assert result.excess_kurtosis > 1.0


# ------------------------------------------------------------- 4. heterogeneity


def test_uniform_grain_has_a_flat_envelope() -> None:
    frames, selected = _material()
    assert evidence.heterogeneity_evidence(frames, selected).envelope_ratio < 0.15


def test_screen_anchored_pattern_is_caught_by_cross_source_comparison() -> None:
    """The check the legacy path could not make, because it never compared across sources.

    A pattern fixed in screen coordinates cancels exactly under temporal differencing, so no
    temporal method can see it at any amplitude. It shows up as correlation between the
    low-frequency residue of two sources whose *pictures* are unrelated.
    """
    frames, selected = _material()
    ys, xs = np.mgrid[0 : frames.shape[1], 0 : frames.shape[2]]
    envelope = 0.06 * (np.sin(xs / 37.0) * np.cos(ys / 41.0))

    other, _ = _material(seed=77, base_level=0.5)
    dirty = frames + envelope[None, :, :]
    dirty_other = other + envelope[None, :, :]

    anchored = evidence.heterogeneity_evidence(dirty, selected, other_source=dirty_other)
    clean = evidence.heterogeneity_evidence(frames, selected, other_source=other)

    assert anchored.is_screen_anchored is True
    assert anchored.belongs_in_a_negative_model is False
    assert clean.is_screen_anchored is False
    assert clean.belongs_in_a_negative_model is True


def test_without_a_second_source_screen_anchoring_is_unknown_not_assumed() -> None:
    frames, selected = _material()
    result = evidence.heterogeneity_evidence(frames, selected)
    assert result.screen_anchored_correlation is None
    assert result.is_screen_anchored is None
    assert result.belongs_in_a_negative_model is None


# ----------------------------------------------------------------- 5. temporal


def test_independence_is_established_only_on_clean_material() -> None:
    frames, selected = _material(rho=0.0)
    result = evidence.temporal_evidence(frames, selected)
    assert result.rho == pytest.approx(0.0, abs=0.05)
    assert result.trusted_fraction == 1.0
    assert result.independence_established


def test_restoration_correlation_denies_independence() -> None:
    """The scene-005 concern: positive rho from denoising must not read as independent."""
    frames, selected = _material(rho=0.5)
    result = evidence.temporal_evidence(frames, selected)
    assert result.rho == pytest.approx(0.5, abs=0.05)
    assert not result.independence_established


def test_a_drifting_window_set_yields_nothing_rather_than_a_wrong_correlation() -> None:
    """Drift masks real correlation, so such windows are rejected before measurement."""
    frames = synthetic.sequence(
        synthetic.SyntheticSpec(
            frames=16,
            height=256,
            width=256,
            sigma=TRUE_SIGMA,
            rho=0.5,
            texture_amplitude=0.05,
            drift_px_per_frame=(0.7, 0.0),
            seed=9,
        )
    )
    report = windows.select_windows(frames, size=96)
    assert not report.accepted
    assert not report.sufficient


# ------------------------------------------------- 5. colour limitation, by type


@pytest.mark.parametrize(
    ("pixel_format", "expected"),
    [
        ("yuv420p10le", chroma.ChromaSupport.LUMA_ONLY),
        ("yuv422p10le", chroma.ChromaSupport.LUMA_ONLY),
        ("nv12", chroma.ChromaSupport.LUMA_ONLY),
        ("gray", chroma.ChromaSupport.LUMA_ONLY),
        ("yuv444p16le", chroma.ChromaSupport.FULL),
        ("rgb48le", chroma.ChromaSupport.FULL),
        ("gbrp16le", chroma.ChromaSupport.FULL),
        ("", chroma.ChromaSupport.UNKNOWN),
        ("something_new", chroma.ChromaSupport.UNKNOWN),
    ],
)
def test_chroma_support_is_classified_from_the_source_format(
    pixel_format: str, expected: chroma.ChromaSupport
) -> None:
    assert chroma.chroma_support_of(pixel_format) is expected


def test_unknown_formats_are_treated_as_luma_only() -> None:
    """Guessing permissively is how an upsampled residual becomes a measured correlation."""
    assert not chroma.chroma_support_of("").supports_colour
    assert not chroma.chroma_support_of("mystery").supports_colour


def test_a_luma_profile_cannot_express_a_colour_claim() -> None:
    """Not a missing field to fill in later: the absence *is* the representation."""
    profile = chroma.profile_for(source_identity="abc", pixel_format="yuv420p10le")
    assert not profile.claims_colour
    assert not hasattr(profile, "channel_covariance")
    assert "colour_limitation" in profile.as_record()


def test_a_colour_profile_cannot_be_built_from_subsampled_material() -> None:
    with pytest.raises(SelectionError, match="cannot be built from luma_only"):
        chroma.ColourGrainProfile(
            source_identity="abc",
            chroma_support=chroma.ChromaSupport.LUMA_ONLY,
            channel_covariance=np.eye(3),
            evidence_basis="scan_444",
        )


def test_a_colour_profile_must_name_how_colour_was_established() -> None:
    with pytest.raises(SelectionError, match="must name"):
        chroma.ColourGrainProfile(
            source_identity="abc",
            chroma_support=chroma.ChromaSupport.FULL,
            channel_covariance=np.eye(3),
            evidence_basis="",
        )


def test_a_colour_profile_is_allowed_from_full_chroma_material() -> None:
    covariance = np.asarray([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
    profile = chroma.ColourGrainProfile(
        source_identity="scan-01",
        chroma_support=chroma.ChromaSupport.FULL,
        channel_covariance=covariance,
        evidence_basis="scan_444",
    )
    assert profile.claims_colour
    assert profile.as_record()["evidence_basis"] == "scan_444"


def test_an_asymmetric_covariance_is_refused() -> None:
    with pytest.raises(SelectionError, match="symmetric"):
        chroma.ColourGrainProfile(
            source_identity="scan-01",
            chroma_support=chroma.ChromaSupport.FULL,
            channel_covariance=np.asarray([[1.0, 0.4, 0.2], [0.9, 1.0, 0.3], [0.2, 0.3, 1.0]]),
            evidence_basis="scan_444",
        )


def test_profile_for_never_infers_colour_from_a_decoded_format() -> None:
    """rgb48le is the *decode output*. Three channels after upsampling prove nothing."""
    profile = chroma.profile_for(source_identity="abc", pixel_format="yuv420p10le")
    assert profile.chroma_support is chroma.ChromaSupport.LUMA_ONLY
    assert not profile.claims_colour
