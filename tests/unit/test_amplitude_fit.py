"""The amplitude-vs-level fit: known-answer recovery, honest validation, model choice."""

from __future__ import annotations

import numpy as np
import pytest

from film_analysis_tools.capabilities.fit import amplitude as amp
from film_analysis_tools.core.errors import DataError, SelectionError


def _power_points(
    a: float,
    b: float,
    sigma0: float = 0.0,
    *,
    intervals: int = 5,
    per: int = 20,
    seed: int = 0,
    between_interval: float = 0.0,
) -> list[amp.AmplitudePoint]:
    """Points drawn from sqrt(sigma0^2 + (a L^b)^2) with a little log-normal scatter.

    ``between_interval`` gives each interval its own amplitude offset, which is what
    leave-one-interval-out is supposed to feel and a tile split would hide.
    """
    rng = np.random.default_rng(seed)
    points: list[amp.AmplitudePoint] = []
    for interval in range(intervals):
        offset = np.exp(rng.normal(0.0, between_interval))
        levels = np.exp(rng.uniform(np.log(2e-4), np.log(0.18), per))
        true = np.sqrt(sigma0**2 + (offset * a * levels**b) ** 2)
        noisy = true * np.exp(rng.normal(0.0, 0.08, per))
        points += [
            amp.AmplitudePoint(level=float(x), sigma=float(y), interval=f"iv{interval}")
            for x, y in zip(levels, noisy, strict=True)
        ]
    return points


def test_a_known_power_law_is_recovered() -> None:
    model = amp.fit_power_floor(_power_points(a=0.06, b=0.73))
    assert model.params["a"] == pytest.approx(0.06, rel=0.25)
    assert model.params["b"] == pytest.approx(0.73, abs=0.1)


def test_the_power_law_beats_the_constant_null_held_out() -> None:
    comparison = amp.compare_models(_power_points(a=0.06, b=0.73))
    assert comparison.power_floor.held_out_log_rmse < comparison.constant.held_out_log_rmse
    assert comparison.chosen.name == "power_floor"


def test_the_compact_model_is_kept_when_the_curve_is_not_materially_better() -> None:
    """More parameters must earn their place on held-out error, not in-sample error."""
    comparison = amp.compare_models(_power_points(a=0.06, b=0.73))
    assert comparison.piecewise.in_sample_log_rmse <= comparison.power_floor.in_sample_log_rmse
    # ... yet the compact model is chosen, because the curve does not improve held out by enough.
    assert comparison.chosen is comparison.power_floor


def test_intervals_are_weighted_equally() -> None:
    """One 200-point interval must not outvote four 5-point intervals."""
    balanced = _power_points(a=0.06, b=0.73, intervals=5, per=20, seed=1)
    lopsided = [
        *(_power_points(a=0.06, b=0.73, intervals=1, per=200, seed=2)),
        *[
            amp.AmplitudePoint(p.level, p.sigma, f"small{i}")
            for i, p in enumerate(_power_points(a=0.2, b=0.4, intervals=1, per=5, seed=3))
        ],
    ]
    # The small divergent intervals still move the fit, because they are weighted as whole units.
    fitted = amp.fit_power_floor(lopsided)
    assert 0.0 < fitted.params["b"] < 0.73, "the minority intervals pulled the exponent down"
    assert amp.fit_power_floor(balanced).params["b"] == pytest.approx(0.73, abs=0.12)


def test_leave_one_interval_out_feels_between_interval_variation() -> None:
    """A random tile split would put near-identical frames in train and test and hide any
    interval-to-interval difference. The interval split exposes it: with real between-interval
    variation the held-out error is meaningfully above the in-sample error."""
    varied = _power_points(a=0.06, b=0.73, between_interval=0.4, per=30)
    model = amp.fit_power_floor(varied)
    assert model.held_out_log_rmse > model.in_sample_log_rmse * 1.1


def test_a_resolved_floor_is_found_when_one_exists() -> None:
    model = amp.fit_power_floor(_power_points(a=0.06, b=0.73, sigma0=0.004))
    assert isinstance(model, amp._PowerFloor)
    assert model.floor_resolved
    assert model.floor_crossover > 0.0


def test_no_floor_is_claimed_for_a_pure_power_law() -> None:
    """The Pulp case: sigma follows the power law to the bottom, so a crossover of 0 would be
    false precision."""
    model = amp.fit_power_floor(_power_points(a=0.06, b=0.73, sigma0=0.0))
    assert isinstance(model, amp._PowerFloor)
    assert not model.floor_resolved
    assert model.floor_crossover == 0.0
    assert (
        "no amplitude floor"
        in amp.compare_models(_power_points(a=0.06, b=0.73, sigma0=0.0)).summary()
    )


def test_the_supported_range_is_reported_and_not_extrapolated() -> None:
    points = _power_points(a=0.06, b=0.73)
    low, high = amp.fit_power_floor(points).supported_level
    assert low == min(p.level for p in points)
    assert high == max(p.level for p in points)
    assert high < 0.25, "there is no highlight data; the range must say so"


def test_too_few_points_or_intervals_is_refused() -> None:
    with pytest.raises(DataError, match="at least 4 points"):
        amp.compare_models(_power_points(a=0.06, b=0.73)[:3])
    one_interval = [amp.AmplitudePoint(0.1, 0.01, "only") for _ in range(6)]
    with pytest.raises(DataError, match="at least two intervals"):
        amp.compare_models(one_interval)


# -------------------------------------------------------------- out-of-range policy


def _fitted() -> amp.FittedModel:
    return amp.fit_power_floor(_power_points(a=0.06, b=0.73))


def test_clamp_holds_the_envelope_at_the_measured_endpoints() -> None:
    """The default. There is no highlight evidence above 0.177, so the safe behaviour is to hold
    the last measured value rather than extrapolate a power law into a region never observed."""
    model = _fitted()
    _, high = model.supported_level
    assert model.predict(1.0, outside="clamp") == pytest.approx(model.predict(high))
    assert model.predict(1e-9, outside="clamp") == pytest.approx(
        model.predict(model.supported_level[0])
    )


def test_extrapolation_is_opt_in_and_differs_from_clamp() -> None:
    model = _fitted()
    extrapolated = float(model.predict(1.0, outside="extrapolate"))
    clamped = float(model.predict(1.0, outside="clamp"))
    assert extrapolated > clamped, "the power law keeps rising above the range"


def test_error_policy_refuses_to_guess() -> None:
    model = _fitted()
    with pytest.raises(SelectionError, match="outside the supported range"):
        model.predict(1.0, outside="error")
    # in range is fine
    mid = sum(model.supported_level) / 2
    assert model.predict(mid, outside="error") > 0


def test_an_unknown_policy_is_refused() -> None:
    with pytest.raises(SelectionError, match="unknown out-of-range policy"):
        _fitted().predict(0.1, outside="wing-it")


def test_the_piecewise_and_power_models_now_agree_on_policy() -> None:
    """The defect this closes: the power model extrapolated while the piecewise model clamped, so
    a consumer got different out-of-range behaviour by accident."""
    points = _power_points(a=0.06, b=0.73)
    power = amp.fit_power_floor(points)
    piecewise = amp.fit_piecewise(points)
    _, high = power.supported_level
    for model in (power, piecewise):
        assert model.predict(2.0, outside="clamp") == pytest.approx(model.predict(high), rel=1e-6)
