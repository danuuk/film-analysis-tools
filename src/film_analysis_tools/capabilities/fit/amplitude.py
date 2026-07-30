"""Amplitude against level: a compact sigma(L) fit with interval-held-out validation.

This fits the **appearance envelope** -- decoded linear-light luma sigma as a function of level --
from trustworthy amplitude points. It is deliberately narrow, and several things it does *not*
attempt are as load-bearing as the fit itself:

* it is not a negative-density granularity curve. This is a final-reference output-luma envelope;
  mapping it into density space is a later, separate step (see the module docstring in the study).
* it says nothing about colour, layer covariance, highlight behaviour beyond the measured range,
  the non-Gaussian shadow tails, or the spatial footprint. One question, answered on its own.

Three decisions keep the fit honest on this data:

**The interval is the independent unit, not the tile.** The 111 points come from 5 intervals, and
tiles inside one interval share nearly identical frames — a random tile split would leak the
validation set into training. Validation is therefore **leave-one-interval-out**.

**Intervals are weighted equally.** One interval contributed 33 points and another 2; without
per-interval weighting the fit would be a fit to whichever scene happened to yield the most tiles.

**Everything is done in log-amplitude coordinates**, because sigma spans from ~1e-4 to ~1.4e-2 and a
linear-residual fit would see only the largest values.

A more complex model is adopted only when its *held-out* error materially improves on the compact
one — never on in-sample error, which always favours more parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from film_analysis_tools.core.errors import DataError, SelectionError

EPS = 1.0e-12

#: A held-out log-RMSE improvement smaller than this is not worth extra parameters.
MATERIAL_IMPROVEMENT = 0.05


@dataclass(frozen=True)
class AmplitudePoint:
    """One trustworthy amplitude measurement, tagged with the interval it came from."""

    level: float
    sigma: float
    interval: str


def _weights(points: Sequence[AmplitudePoint]) -> np.ndarray:
    """One over the count of points in each point's interval, so intervals weigh equally."""
    counts: dict[str, int] = {}
    for point in points:
        counts[point.interval] = counts.get(point.interval, 0) + 1
    return np.array([1.0 / counts[point.interval] for point in points])


def _log_rmse(predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray) -> float:
    residual = np.log(np.maximum(predicted, EPS)) - np.log(np.maximum(observed, EPS))
    return float(np.sqrt(np.sum(weights * residual**2) / max(np.sum(weights), EPS)))


@dataclass(frozen=True)
class FittedModel:
    """A fitted sigma(level), its parameters, and how it did held out.

    ``held_out_log_rmse`` is the honest number: the mean over leave-one-interval-out folds. The
    in-sample figure is reported too, but only so the gap between them is visible.
    """

    name: str
    params: dict[str, float]
    parameter_count: int
    in_sample_log_rmse: float
    held_out_log_rmse: float
    supported_level: tuple[float, float]

    def _predict_raw(self, levels: np.ndarray) -> np.ndarray:
        raise NotImplementedError  # each model implements its own formula

    def predict(self, level: np.ndarray | float, *, outside: str = "clamp") -> np.ndarray:
        """sigma at one or more levels, with an **explicit** out-of-range policy.

        The models disagree outside the supported range — the power law extrapolates, the
        piecewise curve clamps — so a consumer must not get one behaviour by accident. Extrapolation
        is opt-in:

        * ``"clamp"`` (default): levels are clipped to the supported range before evaluation, so the
          envelope is held flat at its measured endpoints. Safe, and there is no highlight evidence
          above 0.177 to extrapolate from.
        * ``"extrapolate"``: evaluate the formula as-is. The caller is asserting the model holds
          beyond where it was measured.
        * ``"error"``: raise on any level outside the range — for a compiler that must refuse to
          guess.
        """
        levels = np.asarray(level, dtype=np.float64)
        low, high = self.supported_level
        if outside == "error":
            if np.any(levels < low) or np.any(levels > high):
                raise SelectionError(
                    f"level outside the supported range [{low:.5f}, {high:.5f}] and outside="
                    "'error'; pass outside='clamp' or 'extrapolate' to state a policy"
                )
        elif outside == "clamp":
            levels = np.clip(levels, low, high)
        elif outside != "extrapolate":
            raise SelectionError(
                f"unknown out-of-range policy {outside!r}; expected clamp, extrapolate or error"
            )
        return self._predict_raw(levels)

    def as_record(self) -> dict[str, Any]:
        low, high = self.supported_level
        return {
            "model": self.name,
            "params": self.params,
            "parameter_count": self.parameter_count,
            "in_sample_log_rmse": self.in_sample_log_rmse,
            "held_out_log_rmse": self.held_out_log_rmse,
            "supported_level_min": low,
            "supported_level_max": high,
            **(
                {
                    "floor_resolved": self.floor_resolved,
                    "floor_crossover": self.floor_crossover,
                }
                if isinstance(self, _PowerFloor)
                else {}
            ),
        }


@dataclass(frozen=True)
class _Constant(FittedModel):
    def _predict_raw(self, levels: np.ndarray) -> np.ndarray:
        return np.full_like(levels, self.params["sigma0"])


@dataclass(frozen=True)
class _PowerFloor(FittedModel):
    def _predict_raw(self, levels: np.ndarray) -> np.ndarray:
        s0, a, b = self.params["sigma0"], self.params["a"], self.params["b"]
        return np.sqrt(s0**2 + (a * np.maximum(levels, 0.0) ** b) ** 2)

    @property
    def floor_crossover(self) -> float:
        """Level where the power term equals the floor: ``(sigma0 / a) ** (1/b)``.

        A *derived*, non-arbitrary boundary for the deepest-shadow region: below it the floor
        dominates and sigma is quantisation- or floor-limited rather than grain-limited. Zero when
        no floor is resolved (see :attr:`floor_resolved`).
        """
        s0, a, b = self.params["sigma0"], self.params["a"], self.params["b"]
        if a <= EPS or b <= EPS or not self.floor_resolved:
            return 0.0
        return float((s0 / a) ** (1.0 / b))

    @property
    def floor_resolved(self) -> bool:
        """Whether a floor is actually resolved, or the model reduced to a pure power law.

        On Pulp Fiction sigma0 collapsed to the grid floor: the amplitude follows the same power
        law all the way down to level 0.00016, with no amplitude floor inside the measured range.
        Reporting a crossover of 0.00000 as though it meant something would be false precision.
        Note this is a statement about *amplitude* only -- the shadow distribution is still
        heavy-tailed there, but that is a shape effect, not a floor in sigma.

        "Resolved" means the floor changes the prediction at the lowest measured level by more than
        10%; a smaller floor is indistinguishable from zero under the measurement scatter.
        """
        s0 = self.params["sigma0"]
        smallest_signal = self.params["a"] * self.supported_level[0] ** self.params["b"]
        if smallest_signal <= EPS:
            return s0 > EPS
        prediction_lift = float(np.sqrt(1.0 + (s0 / smallest_signal) ** 2))
        return prediction_lift > 1.10


@dataclass(frozen=True)
class _Piecewise(FittedModel):
    knot_log_levels: tuple[float, ...] = ()
    knot_log_sigmas: tuple[float, ...] = ()

    def _predict_raw(self, levels: np.ndarray) -> np.ndarray:
        log_levels = np.log(np.maximum(levels, EPS))
        interpolated = np.interp(log_levels, self.knot_log_levels, self.knot_log_sigmas)
        return np.exp(interpolated)


# --------------------------------------------------------------------------- fitters


def _fit_constant_params(points: Sequence[AmplitudePoint], weights: np.ndarray) -> dict[str, float]:
    log_sigma = np.log(np.array([max(p.sigma, EPS) for p in points]))
    return {"sigma0": float(np.exp(np.sum(weights * log_sigma) / np.sum(weights)))}


def _fit_power_floor_params(
    points: Sequence[AmplitudePoint], weights: np.ndarray
) -> dict[str, float]:
    """Coarse-to-fine grid over (sigma0, a, b), minimising weighted log-residual.

    A grid rather than a solver because scipy is not a dependency and the parameter space is three
    numbers — but the grid is refined twice around the best cell, so it is not coarse in the answer.
    """
    levels = np.array([max(p.level, EPS) for p in points])
    sigmas = np.array([max(p.sigma, EPS) for p in points])

    def evaluate(s0: float, a: float, b: float) -> float:
        predicted = np.sqrt(s0**2 + (a * levels**b) ** 2)
        return _log_rmse(predicted, sigmas, weights)

    log_s0 = np.log10([sigmas.min() * 0.02, sigmas.max()])
    log_a = np.array([-3.0, 1.5])
    b_range = np.array([0.1, 2.5])
    best = {"sigma0": float(sigmas.min()), "a": 0.1, "b": 1.0}
    best_error = np.inf
    for _ in range(3):
        for s0 in np.logspace(log_s0[0], log_s0[1], 20):
            for a in np.logspace(log_a[0], log_a[1], 20):
                for b in np.linspace(b_range[0], b_range[1], 20):
                    error = evaluate(s0, a, b)
                    if error < best_error:
                        best_error, best = error, {"sigma0": s0, "a": a, "b": b}
        # Refine the search window around the winner.
        log_s0 = np.log10(best["sigma0"]) + np.array([-0.4, 0.4])
        log_a = np.log10(max(best["a"], 1e-3)) + np.array([-0.5, 0.5])
        b_range = best["b"] + np.array([-0.3, 0.3])
    return {key: float(value) for key, value in best.items()}


def _fit_piecewise_params(
    points: Sequence[AmplitudePoint], weights: np.ndarray, knots: int
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear in log-log: knot heights by weighted least squares at fixed knot levels."""
    log_level = np.log(np.array([max(p.level, EPS) for p in points]))
    log_sigma = np.log(np.array([max(p.sigma, EPS) for p in points]))
    knot_levels = np.quantile(log_level, np.linspace(0.0, 1.0, knots))
    knot_levels = np.unique(knot_levels)
    if knot_levels.size < 2:
        raise DataError("piecewise fit needs at least two distinct knot levels")

    # Hat-function basis: each column is the contribution of one knot to np.interp.
    basis = np.zeros((log_level.size, knot_levels.size))
    for column in range(knot_levels.size):
        unit = np.zeros(knot_levels.size)
        unit[column] = 1.0
        basis[:, column] = np.interp(log_level, knot_levels, unit)

    sqrt_w = np.sqrt(weights)[:, None]
    heights, *_ = np.linalg.lstsq(basis * sqrt_w, log_sigma * np.sqrt(weights), rcond=None)
    return knot_levels, heights


# ------------------------------------------------------------------- held-out validation


def _held_out_log_rmse(
    points: Sequence[AmplitudePoint],
    predict_after_fit: Any,
) -> float:
    """Mean leave-one-interval-out log-RMSE, each held-out interval weighted equally."""
    intervals = sorted({point.interval for point in points})
    if len(intervals) < 2:
        return float("nan")
    errors: list[float] = []
    for held in intervals:
        train = [p for p in points if p.interval != held]
        test = [p for p in points if p.interval == held]
        if not train or not test:
            continue
        predictor = predict_after_fit(train)
        predicted = predictor(np.array([p.level for p in test]))
        observed = np.array([p.sigma for p in test])
        errors.append(_log_rmse(predicted, observed, np.ones(len(test))))
    return float(np.mean(errors)) if errors else float("nan")


def _supported_range(points: Sequence[AmplitudePoint]) -> tuple[float, float]:
    levels = [p.level for p in points]
    return (min(levels), max(levels))


def fit_constant(points: Sequence[AmplitudePoint]) -> FittedModel:
    weights = _weights(points)
    params = _fit_constant_params(points, weights)

    def refit(train: Sequence[AmplitudePoint]) -> Any:
        local = _fit_constant_params(train, _weights(train))
        return lambda level: np.full_like(np.asarray(level, dtype=np.float64), local["sigma0"])

    model = _Constant(
        name="constant",
        params=params,
        parameter_count=1,
        in_sample_log_rmse=_log_rmse(
            np.full(len(points), params["sigma0"]),
            np.array([p.sigma for p in points]),
            weights,
        ),
        held_out_log_rmse=_held_out_log_rmse(points, refit),
        supported_level=_supported_range(points),
    )
    return model


def fit_power_floor(points: Sequence[AmplitudePoint]) -> FittedModel:
    weights = _weights(points)
    params = _fit_power_floor_params(points, weights)

    def predict(level: np.ndarray | float, p: dict[str, float]) -> np.ndarray:
        levels = np.asarray(level, dtype=np.float64)
        return np.sqrt(p["sigma0"] ** 2 + (p["a"] * np.maximum(levels, 0.0) ** p["b"]) ** 2)

    def refit(train: Sequence[AmplitudePoint]) -> Any:
        local = _fit_power_floor_params(train, _weights(train))
        return lambda level: predict(level, local)

    return _PowerFloor(
        name="power_floor",
        params=params,
        parameter_count=3,
        in_sample_log_rmse=_log_rmse(
            predict(np.array([p.level for p in points]), params),
            np.array([p.sigma for p in points]),
            weights,
        ),
        held_out_log_rmse=_held_out_log_rmse(points, refit),
        supported_level=_supported_range(points),
    )


def fit_piecewise(points: Sequence[AmplitudePoint], *, knots: int = 4) -> FittedModel:
    weights = _weights(points)
    knot_levels, heights = _fit_piecewise_params(points, weights, knots)

    def predict(level: np.ndarray | float, kl: np.ndarray, kh: np.ndarray) -> np.ndarray:
        log_level = np.log(np.maximum(np.asarray(level, dtype=np.float64), EPS))
        return np.exp(np.interp(log_level, kl, kh))

    def refit(train: Sequence[AmplitudePoint]) -> Any:
        kl, kh = _fit_piecewise_params(train, _weights(train), knots)
        return lambda level: predict(level, kl, kh)

    return _Piecewise(
        name=f"piecewise_{knot_levels.size}knot",
        params={f"knot{i}": float(h) for i, h in enumerate(heights)},
        parameter_count=int(knot_levels.size),
        in_sample_log_rmse=_log_rmse(
            predict(np.array([p.level for p in points]), knot_levels, heights),
            np.array([p.sigma for p in points]),
            weights,
        ),
        held_out_log_rmse=_held_out_log_rmse(points, refit),
        supported_level=_supported_range(points),
        knot_log_levels=tuple(float(x) for x in knot_levels),
        knot_log_sigmas=tuple(float(x) for x in heights),
    )


# --------------------------------------------------------------------------- comparison


@dataclass(frozen=True)
class ModelComparison:
    """The three candidate models, and which one the held-out error supports."""

    constant: FittedModel
    power_floor: FittedModel
    piecewise: FittedModel
    intervals: int
    points: int

    @property
    def ranked(self) -> list[FittedModel]:
        return sorted(
            (self.constant, self.power_floor, self.piecewise),
            key=lambda m: m.held_out_log_rmse,
        )

    @property
    def chosen(self) -> FittedModel:
        """The simplest model within :data:`MATERIAL_IMPROVEMENT` of the best held-out error.

        More parameters must earn their place on *held-out* error. The compact power/floor model
        is preferred over the piecewise curve unless the curve is materially better held out.
        """
        best = self.ranked[0]
        candidates = [
            m
            for m in (self.constant, self.power_floor, self.piecewise)
            if m.held_out_log_rmse <= best.held_out_log_rmse + MATERIAL_IMPROVEMENT
        ]
        return min(candidates, key=lambda m: m.parameter_count)

    def as_record(self) -> dict[str, Any]:
        return {
            "intervals": self.intervals,
            "points": self.points,
            "validation": "leave-one-interval-out",
            "chosen": self.chosen.name,
            "models": [
                self.constant.as_record(),
                self.power_floor.as_record(),
                self.piecewise.as_record(),
            ],
        }

    def summary(self) -> str:
        lines = [
            f"amplitude vs level — {self.points} points, {self.intervals} intervals "
            "(leave-one-interval-out)",
            f"  {'model':<18}{'params':>7}{'in-sample':>12}{'held-out':>11}",
        ]
        for model in (self.constant, self.power_floor, self.piecewise):
            mark = "  <- chosen" if model is self.chosen else ""
            lines.append(
                f"  {model.name:<18}{model.parameter_count:>7}"
                f"{model.in_sample_log_rmse:>12.4f}{model.held_out_log_rmse:>11.4f}{mark}"
            )
        low, high = self.chosen.supported_level
        lines.append(f"  supported level range: {low:.5f} .. {high:.5f} linear (no highlight data)")
        if isinstance(self.chosen, _PowerFloor):
            params = self.chosen.params
            lines.append(
                f"  sigma(L) = sqrt({params['sigma0']:.5f}^2 + "
                f"({params['a']:.4f} L^{params['b']:.3f})^2)"
            )
            if self.chosen.floor_resolved:
                lines.append(
                    f"  floor crossover at level {self.chosen.floor_crossover:.5f}: below it the "
                    "measurement is floor/quantisation-limited, not grain-limited"
                )
            else:
                lines.append(
                    "  no amplitude floor resolved: a pure power law describes sigma across the "
                    "whole measured range (the shadow tails are a shape effect, not a sigma floor)"
                )
        return "\n".join(lines)


def compare_models(points: Sequence[AmplitudePoint]) -> ModelComparison:
    """Fit and cross-validate all three candidates."""
    if len(points) < 4:
        raise DataError(f"need at least 4 points to compare amplitude models; got {len(points)}")
    intervals = len({point.interval for point in points})
    if intervals < 2:
        raise DataError("leave-one-interval-out needs at least two intervals")
    return ModelComparison(
        constant=fit_constant(points),
        power_floor=fit_power_floor(points),
        piecewise=fit_piecewise(points),
        intervals=intervals,
        points=len(points),
    )


__all__ = [
    "MATERIAL_IMPROVEMENT",
    "AmplitudePoint",
    "FittedModel",
    "ModelComparison",
    "compare_models",
    "fit_constant",
    "fit_piecewise",
    "fit_power_floor",
]
