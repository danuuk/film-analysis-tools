"""Residual extraction, checked against synthetic grain with known sigma and known rho.

This is the known-answer harness the legacy grain path never had. Its absence is how a systematic
amplitude bias survived: with only real footage to test against, there is no true value, and a
plausible number is indistinguishable from a correct one.

Every threshold in the extractor was chosen or corrected because one of these tests failed first.
"""

from __future__ import annotations

import numpy as np
import pytest

from film_analysis_tools.capabilities.measure import residual, synthetic
from film_analysis_tools.capabilities.measure.residual import (
    SUBPIXEL_REJECT as SUBPIXEL_REJECT_LOCAL,
)
from film_analysis_tools.core.errors import DataError, SelectionError

TOLERANCE = 0.03  # 3% on sigma
RHO_TOLERANCE = 0.03


def _spec(**kwargs: object) -> synthetic.SyntheticSpec:
    base: dict[str, object] = {"frames": 24, "height": 160, "width": 160, "seed": 7}
    base.update(kwargs)
    return synthetic.SyntheticSpec(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ the generator itself


def test_generated_grain_has_the_declared_sigma() -> None:
    """If the harness is wrong, everything it certifies is wrong."""
    spec = _spec(sigma=0.02, rho=0.5)
    fields = synthetic.grain_fields(spec)
    assert float(np.std(fields)) == pytest.approx(0.02, rel=0.03)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9, -0.4])
def test_generated_grain_has_the_declared_correlation(rho: float) -> None:
    fields = synthetic.grain_fields(_spec(frames=200, height=64, width=64, sigma=0.01, rho=rho))
    flat = fields.reshape(fields.shape[0], -1)
    observed = float(np.mean(np.sum(flat[:-1] * flat[1:], axis=1) / np.sum(flat[:-1] ** 2, axis=1)))
    assert observed == pytest.approx(rho, abs=0.05)


def test_difference_variance_matches_the_closed_form() -> None:
    """``Var(g_t - g_t-k) = 2 sigma^2 (1 - rho^k)`` — the identity the estimator inverts."""
    spec = _spec(frames=400, height=64, width=64, sigma=0.01, rho=0.5)
    fields = synthetic.grain_fields(spec)
    for lag in (1, 2, 4):
        observed = float(np.var(fields[lag:] - fields[:-lag]))
        assert observed == pytest.approx(spec.expected_difference_variance(lag), rel=0.06)


def test_the_spec_refuses_impossible_parameters() -> None:
    with pytest.raises(SelectionError):
        synthetic.SyntheticSpec(rho=1.5)
    with pytest.raises(SelectionError):
        synthetic.SyntheticSpec(sigma=-1.0)


# --------------------------------------------------------- recovery on featureless windows


@pytest.mark.parametrize(
    ("sigma", "rho"),
    [
        (0.01, 0.0),
        (0.01, 0.3),
        (0.01, 0.6),
        (0.02, 0.45),
        (0.008, 0.8),
        (0.005, -0.3),
        (0.01, -0.5),
    ],
)
def test_recovers_known_sigma_and_rho(sigma: float, rho: float) -> None:
    estimate = residual.extract(synthetic.sequence(_spec(sigma=sigma, rho=rho)))
    assert estimate.sigma == pytest.approx(sigma, rel=TOLERANCE)
    assert estimate.rho == pytest.approx(rho, abs=RHO_TOLERANCE)


def test_lag4_agrees_with_lag1_and_lag2() -> None:
    """Independent estimate. Disagreement means AR(1) does not describe the window."""
    estimate = residual.extract(synthetic.sequence(_spec(sigma=0.01, rho=0.6)))
    assert estimate.correlation_consistent
    assert estimate.rho_from_lag4 == pytest.approx(0.6, abs=0.05)


def test_the_legacy_estimator_is_the_rho_zero_corner() -> None:
    """At zero correlation the corrected estimate must agree with ``sqrt(var/2)``."""
    estimate = residual.extract(synthetic.sequence(_spec(sigma=0.01, rho=0.0)))
    assert estimate.sigma == pytest.approx(estimate.legacy_sigma, rel=0.01)
    assert estimate.correlation_correction == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize(("rho", "min_shortfall"), [(0.3, 0.10), (0.6, 0.30), (0.8, 0.45)])
def test_quantifies_how_far_the_legacy_estimator_understates(
    rho: float, min_shortfall: float
) -> None:
    """The bias the legacy path never measured, in the direction the algebra predicts.

    ``sqrt(var/2)`` understates sigma by ``sqrt(1 - rho)`` whenever grain persists across frames.
    """
    sigma = 0.01
    estimate = residual.extract(synthetic.sequence(_spec(sigma=sigma, rho=rho)))
    shortfall = (sigma - estimate.legacy_sigma) / sigma
    assert shortfall >= min_shortfall
    assert estimate.legacy_sigma == pytest.approx(sigma * np.sqrt(1.0 - rho), rel=0.05)


def test_negative_correlation_makes_the_legacy_estimator_overstate() -> None:
    """The legacy gate was one-sided; codec inter-frame filtering can invert the sign."""
    sigma = 0.01
    estimate = residual.extract(synthetic.sequence(_spec(sigma=sigma, rho=-0.5)))
    assert estimate.legacy_sigma > sigma
    assert estimate.sigma == pytest.approx(sigma, rel=TOLERANCE)


# ------------------------------------------------------------------- alignment behaviour


def test_a_static_window_is_not_aligned() -> None:
    """Applying a shift where none is needed destroys the correlation being measured."""
    estimate = residual.extract(
        synthetic.sequence(_spec(sigma=0.01, rho=0.6, texture_amplitude=0.05))
    )
    assert not estimate.alignment_applied
    assert estimate.max_integer_shift == 0
    assert estimate.rho == pytest.approx(0.6, abs=RHO_TOLERANCE)


@pytest.mark.parametrize("drift", [(1.0, 0.0), (0.0, 2.0), (1.0, 1.0), (3.0, 2.0)])
def test_alignment_rescues_a_drifting_textured_window(drift: tuple[float, float]) -> None:
    """The legacy failure, reproduced and fixed.

    Without alignment, drifting texture inflates the residual enormously — this is the mechanism
    behind the legacy preset's warning that one scene's static *textured* residual measured 7.15x
    its grain-dominated residual.
    """
    frames = synthetic.sequence(
        _spec(sigma=0.01, rho=0.0, texture_amplitude=0.06, drift_px_per_frame=drift, seed=5)
    )
    aligned = residual.extract(frames, align=True)
    unaligned = residual.extract(frames, align=False)

    assert aligned.alignment_applied
    assert aligned.sigma == pytest.approx(0.01, rel=TOLERANCE)
    assert unaligned.sigma > aligned.sigma * 2.0


def test_featureless_windows_are_never_aligned_at_any_correlation() -> None:
    """Negative correlation makes zero lag the *worst* cross-correlation, so an ungated aligner
    moves away from it and reports the correlation as near zero."""
    for rho in (-0.5, -0.3, 0.0, 0.5):
        estimate = residual.extract(synthetic.sequence(_spec(sigma=0.01, rho=rho)))
        assert not estimate.alignment_applied, rho
        assert estimate.rho == pytest.approx(rho, abs=RHO_TOLERANCE), rho


def test_shift_estimation_recovers_a_known_translation() -> None:
    frames = synthetic.sequence(_spec(frames=2, sigma=0.01, texture_amplitude=0.06, seed=3))
    rolled = np.roll(frames[1], (2, -3), axis=(0, 1))
    shift = residual.estimate_shift(frames[0], rolled)
    assert (shift.dy, shift.dx) == (-2, 3)
    assert shift.gain > 0.5


def test_structure_ratio_is_one_when_there_is_no_structure() -> None:
    """Scale-invariance is why this ratio is the gate: it needs no per-corpus calibration."""
    for sigma in (0.005, 0.01, 0.02):
        frames = synthetic.sequence(_spec(frames=2, sigma=sigma, texture_amplitude=0.0, seed=3))
        shift = residual.estimate_shift(frames[0], frames[1])
        assert shift.structure_snr == pytest.approx(1.0, abs=0.35), sigma


def test_bound_is_reported_so_a_fast_window_can_be_rejected() -> None:
    """Reject the window rather than relax the search — motion beyond the bound is not static."""
    frames = synthetic.sequence(
        _spec(sigma=0.01, texture_amplitude=0.06, drift_px_per_frame=(4.0, 0.0), seed=5)
    )
    estimate = residual.extract(frames, max_shift=2)
    assert estimate.at_bound
    assert estimate.drifting


# ----------------------------------------------------------------------------- contracts


def test_too_few_frames_is_an_error_naming_the_requirement() -> None:
    frames = synthetic.sequence(_spec(frames=3))
    with pytest.raises(DataError, match="at least 5 frames"):
        residual.extract(frames, lags=(1, 2, 4))


def test_wrong_shape_is_rejected() -> None:
    with pytest.raises(DataError, match="expected"):
        residual.extract(np.zeros((4, 4)))


def test_estimate_serialises_with_its_evidence() -> None:
    record = residual.extract(synthetic.sequence(_spec(sigma=0.01, rho=0.4))).as_record()
    for field in (
        "sigma",
        "rho",
        "legacy_sigma",
        "correlation_correction",
        "rho_from_lag4",
        "correlation_consistent",
        "lag_variances",
        "alignment_applied",
        "at_bound",
        "sample_count",
    ):
        assert field in record, field


# --------------------------------------------------- trust in the reported correlation


def test_correlation_is_trustworthy_on_a_static_window() -> None:
    estimate = residual.extract(synthetic.sequence(_spec(sigma=0.01, rho=0.5)))
    assert estimate.correlation_trustworthy
    assert estimate.rho == pytest.approx(0.5, abs=RHO_TOLERANCE)


def test_drift_masks_real_correlation_and_the_estimate_says_so() -> None:
    """The scene-005 shape: restoration correlation *and* gate weave together.

    Sub-pixel drift decorrelates consecutive frames, so a genuinely correlated window reads back
    as independent — a true rho of 0.5 with 0.7 px of drift measures about 0.05. Amplitude
    survives because the two errors partly cancel; the correlation does not. Reporting
    "temporal independence established" here would be exactly the wrong conclusion, so the
    estimate refuses to vouch for its own rho.
    """
    frames = synthetic.sequence(
        _spec(sigma=0.01, rho=0.5, texture_amplitude=0.05, drift_px_per_frame=(0.7, 0.0), seed=9)
    )
    estimate = residual.extract(frames)
    assert estimate.rho < 0.2, "drift is expected to mask the true correlation"
    assert estimate.drifting
    assert not estimate.correlation_trustworthy


def test_fixed_pattern_noise_is_invisible_to_temporal_differencing() -> None:
    """A pattern identical in every frame contributes nothing to Var(f_t - f_t-1).

    Screen-anchored heterogeneity is therefore not something this method can find, at any
    amplitude — detecting it needs a spatial comparison across sources, not a temporal one.
    Recorded as a test so the limit is not rediscovered as a surprise.
    """
    spec = _spec(sigma=0.01, rho=0.0, seed=9)
    clean = synthetic.sequence(spec)
    ys, xs = np.mgrid[0 : spec.height, 0 : spec.width]
    envelope = 0.06 * (np.sin(xs / 37.0) * np.cos(ys / 41.0))
    assert float(envelope.std()) > 2.0 * spec.sigma

    with_pattern = residual.extract(clean + envelope[None, :, :])
    without = residual.extract(clean)
    assert with_pattern.sigma == pytest.approx(without.sigma, rel=1e-9)


# ---------------------------------------------------- correlation identifiability


def _estimate(**kwargs: float) -> residual.ResidualEstimate:
    base: dict[str, object] = {
        "sigma": 0.01,
        "rho": 0.5,
        "legacy_sigma": 0.01,
        "lag_variances": {1: 1.0},
        "rho_from_lag4": 0.5,
        "subpixel_residual": 0.0,
        "max_integer_shift": 0,
        "structure_snr": 1.0,
        "at_bound": False,
        "structure": 0.0,
        "alignment_applied": False,
        "motion_energy": 0.0,
        "grain_hp_std": 0.01,
        "sample_count": 100,
    }
    base.update(kwargs)
    return residual.ResidualEstimate(**base)  # type: ignore[arg-type]


def test_a_saturated_correlation_is_not_identified() -> None:
    """The failure this exists for.

    rho is clamped to +/-0.99. Two estimates pinned to the same edge agree perfectly, so a
    consistency check alone called them trustworthy — while the amplitude correction
    1/sqrt(1-rho) multiplies sigma by 10 there. On the committed run Sony's six saturated points
    had median sigma 0.00736 against 0.000759 for the nineteen identified ones.
    """
    saturated = _estimate(rho=residual.RHO_BOUND, rho_from_lag4=residual.RHO_BOUND)
    assert saturated.rho_saturated
    assert not saturated.parameter_identified
    assert not saturated.correlation_consistent, "agreement at the clamp is an artefact"
    assert not saturated.correlation_trustworthy


def test_an_interior_correlation_is_identified() -> None:
    inside = _estimate(rho=0.40, rho_from_lag4=0.43)
    assert not inside.rho_saturated
    assert inside.parameter_identified
    assert inside.correlation_trustworthy


def test_the_unclamped_solution_is_kept() -> None:
    """A clamped rho is a lower limit; the raw value says how far outside the model the data is."""
    frames = synthetic.sequence(synthetic.SyntheticSpec(frames=8, sigma=0.01, rho=0.0, seed=5))
    estimate = residual.extract(frames)
    assert estimate.raw_rho == pytest.approx(estimate.rho, abs=1e-6)


# ------------------------------------------------ coherent motion vs registration noise


def test_a_single_unreliable_pair_does_not_reject_a_stationary_tile() -> None:
    """The defect this replaced: rejecting on the *maximum* sub-pixel residual and *any* boundary
    contact meant one bad pair in 47 rejected an otherwise stationary tile."""
    from film_analysis_tools.capabilities.measure.residual import Shift, _shift_statistics

    rng = np.random.default_rng(1)
    # 46 pairs of scattered registration noise (no common direction) plus one large outlier.
    noise = [
        Shift(0, 0, float(dy), float(dx), 0.0, 5.0, 5.0, at_bound=False)
        for dy, dx in rng.normal(0.0, 0.03, (46, 2))
    ]
    outlier = Shift(0, 0, 0.6, 0.5, 0.0, 5.0, 5.0, at_bound=True)
    p90, boundary, coherence, _ = _shift_statistics([*noise, outlier])
    assert p90 < SUBPIXEL_REJECT_LOCAL, "the robust residual ignores the single outlier"
    assert boundary < 0.05, "one boundary hit in 47 is not coherent motion"
    assert coherence < 0.5, "scattered residuals do not describe a direction"


def test_coherent_drift_is_still_rejected() -> None:
    """Every pair agreeing on a direction is real motion, and must still fail the gate."""
    frames = synthetic.sequence(
        _spec(sigma=0.01, rho=0.0, texture_amplitude=0.06, drift_px_per_frame=(0.5, 0.0), frames=48)
    )
    estimate = residual.extract(frames)
    assert estimate.shift_coherence > 0.9
    assert estimate.subpixel_p90 > 0.25
    assert estimate.drifting


def test_a_stationary_textured_tile_is_not_drifting() -> None:
    frames = synthetic.sequence(
        _spec(sigma=0.01, rho=0.0, texture_amplitude=0.06, drift_px_per_frame=(0.0, 0.0), frames=48)
    )
    estimate = residual.extract(frames)
    assert estimate.shift_coherence < 0.5
    assert not estimate.drifting
