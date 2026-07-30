"""Held-out reconstruction: a known footprint round-trips, and the folds stay honest."""

from __future__ import annotations

import numpy as np
import pytest

from film_analysis_tools.capabilities.fit import reconstruct as rc
from film_analysis_tools.capabilities.fit.amplitude import AmplitudePoint
from film_analysis_tools.capabilities.measure import footprint as fp
from film_analysis_tools.core.errors import DataError


def _grain(*, n=16, size=96, correlation_px=1.5, aniso=1.0, amplitude=0.01, seed=0):
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, (n, size, size))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    gauss = np.exp(-2.0 * (np.pi * correlation_px) ** 2 * (fy**2 + (fx * aniso) ** 2))
    field = np.real(np.fft.ifft2(np.fft.fft2(white, axes=(1, 2)) * gauss, axes=(1, 2)))
    field /= field.std()
    return 0.2 + amplitude * field


def _spectra(*, correlation_px=0.6, aniso=1.0, intervals=4, per=5, seed=0):
    out = []
    s = seed
    for iv in range(intervals):
        for _ in range(per):
            s += 1
            frames = _grain(correlation_px=correlation_px, aniso=aniso, seed=s)
            out.append(
                fp.window_spectrum(frames, interval=f"iv{iv}", level=0.1, band="mid", sigma_hat=1.0)
            )
    return out


def _points(intervals=4, per=5, a=0.06, b=0.73, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    for iv in range(intervals):
        for _ in range(per):
            level = float(np.exp(rng.uniform(np.log(2e-4), np.log(0.15))))
            sigma = a * level**b * float(np.exp(rng.normal(0, 0.08)))
            pts.append(AmplitudePoint(level=level, sigma=sigma, interval=f"iv{iv}"))
    return pts


# --------------------------------------------------------- materialisation


def test_a_materialised_filter_regenerates_the_footprint() -> None:
    """sqrt(pooled PSD) driven with white noise must reproduce the measured radial shape."""
    spectra = _spectra(correlation_px=0.6)
    filt = rc.materialise_filter(spectra)
    generated = rc.generate_frames(filt, 40, seed=1)
    gen = fp.window_spectrum(generated, interval="gen", level=0.1, band="mid", sigma_hat=1.0)
    measured = fp._mean_spectrum(spectra)
    assert fp.spectral_distance(gen, measured, direction="radial") < 0.15


def test_the_filter_reproduces_anisotropy() -> None:
    spectra = _spectra(correlation_px=2.0, aniso=2.0)
    generated = rc.generate_frames(rc.materialise_filter(spectra), 20, seed=2)
    gen = fp.window_spectrum(generated, interval="gen", level=0.1, band="mid", sigma_hat=1.0)
    measured = fp._mean_spectrum(spectra)
    assert gen.anisotropy == pytest.approx(measured.anisotropy, rel=0.3)
    assert gen.grain_radius_h > gen.grain_radius_v


def test_each_interval_weighs_equally_in_the_pool() -> None:
    """A window-rich interval must not dominate the kernel: a 3-window interval with a different
    footprint still pulls the pool away from a 40-window one."""
    big = _spectra(correlation_px=0.6, intervals=1, per=40, seed=2)
    small = _spectra(correlation_px=2.5, intervals=1, per=3, seed=3)
    for one in small:
        object.__setattr__(one, "interval", "small")
    pooled = rc.materialise_filter(big + small)
    big_only = rc.materialise_filter(big)
    assert not np.allclose(pooled, big_only, atol=1e-6)


def test_generated_frames_are_gaussian_and_independent() -> None:
    generated = rc.generate_frames(rc.materialise_filter(_spectra()), 30, seed=4)
    residual = np.diff(generated, axis=0).ravel()
    skew, kurt = rc._skew_kurtosis(residual)
    assert abs(skew) < 0.1 and abs(kurt) < 0.2
    assert abs(rc._lag_ratio_rho(generated)) < 0.1


# --------------------------------------------------------- held-out folds


def test_reconstruction_matches_a_held_out_interval() -> None:
    """The whole point: a footprint pooled from four intervals reproduces the fifth it never saw."""
    result = rc.reconstruct(
        _spectra(correlation_px=0.6, intervals=5), _points(intervals=5), frames=40
    )
    assert len(result.folds) == 5
    assert result._spread(lambda f: f.radial_psd_distance)[0] < 0.2
    assert result._spread(lambda f: f.amplitude_log_error)[0] < 0.2
    assert "reconstruction" in result.summary()


def test_the_held_interval_is_never_in_its_own_training() -> None:
    result = rc.reconstruct(_spectra(intervals=4), _points(intervals=4))
    for fold in result.folds:
        assert fold.train_intervals == 3, "one interval held out of four"


def test_reconstruction_needs_two_intervals() -> None:
    with pytest.raises(DataError, match="at least two intervals"):
        rc.reconstruct(_spectra(intervals=1), _points(intervals=1))


def test_a_spectrum_without_a_2d_psd_cannot_materialise() -> None:
    stripped = fp.WindowSpectrum(
        interval="a", level=0.1, band="m", frequencies=(0.1,), radial_psd=(1.0,),
        horizontal_psd=(1.0,), vertical_psd=(1.0,), grain_radius=1.0, grain_radius_h=1.0,
        grain_radius_v=1.0, anisotropy=1.0, block_peak_h=1.0, block_peak_v=1.0,
    )  # fmt: skip
    with pytest.raises(DataError, match="no window carries a 2D PSD"):
        rc.materialise_filter([stripped])
