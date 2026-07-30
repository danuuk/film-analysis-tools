"""Spatial footprint: amplitude-invariance, shape discrimination, the stability decision."""

from __future__ import annotations

import numpy as np
import pytest

from film_analysis_tools.capabilities.measure import footprint as fp
from film_analysis_tools.core.errors import DataError


def _grain(
    *,
    n: int = 12,
    size: int = 96,
    correlation_px: float = 0.0,
    amplitude: float = 0.01,
    aniso: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Frames whose lag-one difference has a known spatial footprint.

    ``correlation_px`` is the Gaussian blur width of the grain (0 = spatially white); ``aniso``
    stretches that width horizontally. Independent grain per frame, on a flat base.
    """
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, (n, size, size))
    if correlation_px <= 0.0:
        field = white
    else:
        fy = np.fft.fftfreq(size)[:, None]
        fx = np.fft.fftfreq(size)[None, :]
        gauss = np.exp(-2.0 * (np.pi * correlation_px) ** 2 * (fy**2 + (fx * aniso) ** 2))
        field = np.real(np.fft.ifft2(np.fft.fft2(white, axes=(1, 2)) * gauss, axes=(1, 2)))
        field /= field.std()
    return 0.2 + amplitude * field


def _spectrum(frames: np.ndarray, *, interval: str, band: str, level: float, sigma: float):
    return fp.window_spectrum(frames, interval=interval, level=level, band=band, sigma_hat=sigma)


# --------------------------------------------------------------- shape measurement


def test_white_grain_reads_as_a_small_isotropic_footprint() -> None:
    spectrum = _spectrum(
        _grain(correlation_px=0.0), interval="a", band="mid", level=0.1, sigma=0.01
    )
    assert spectrum.grain_radius < 1.5, "white grain is one pixel wide"
    assert spectrum.anisotropy == pytest.approx(1.0, abs=0.4)


def test_spatially_correlated_grain_reads_as_a_larger_footprint() -> None:
    white = _spectrum(
        _grain(correlation_px=0.0, seed=1), interval="a", band="m", level=0.1, sigma=0.01
    )
    broad = _spectrum(
        _grain(correlation_px=2.0, seed=1), interval="a", band="m", level=0.1, sigma=0.01
    )
    assert broad.grain_radius > white.grain_radius * 1.5


def test_anisotropic_grain_is_detected() -> None:
    spectrum = _spectrum(
        _grain(correlation_px=2.0, aniso=2.5, seed=2), interval="a", band="m", level=0.1, sigma=0.01
    )
    assert spectrum.anisotropy > 1.3


# ----------------------------------------------------------- amplitude invariance


def test_normalisation_removes_amplitude_so_shape_can_be_compared() -> None:
    """The whole point: the same footprint at two amplitudes must look the same once each is
    divided by its own predicted sigma."""
    quiet = _grain(correlation_px=1.5, amplitude=0.002, seed=3)
    loud = _grain(correlation_px=1.5, amplitude=0.02, seed=4)
    # sigma_hat is the true residual amplitude: amplitude * sqrt(2) (difference of two fields).
    a = _spectrum(quiet, interval="a", band="m", level=0.02, sigma=0.002 * np.sqrt(2))
    b = _spectrum(loud, interval="a", band="m", level=0.20, sigma=0.02 * np.sqrt(2))
    assert fp.spectral_distance(a, b) < 0.25, "same footprint, different amplitude, small distance"


def test_different_footprints_are_far_apart() -> None:
    white = _spectrum(
        _grain(correlation_px=0.0, seed=5), interval="a", band="m", level=0.1, sigma=0.01
    )
    broad = _spectrum(
        _grain(correlation_px=2.5, seed=5), interval="a", band="m", level=0.1, sigma=0.01
    )
    same = _spectrum(
        _grain(correlation_px=0.0, seed=6), interval="a", band="m", level=0.1, sigma=0.01
    )
    assert fp.spectral_distance(white, broad) > 3 * fp.spectral_distance(white, same)


# ------------------------------------------------------------- the stability test


def _corpus(
    footprint_by_band: dict[str, float],
    *,
    aniso_by_band: dict[str, float] | None = None,
    per: int = 6,
    intervals: int = 3,
):
    spectra = []
    seed = 0
    for interval in range(intervals):
        for band, corr in footprint_by_band.items():
            aniso = (aniso_by_band or {}).get(band, 1.0)
            for _ in range(per):
                seed += 1
                frames = _grain(correlation_px=corr, aniso=aniso, seed=seed)
                spectra.append(
                    _spectrum(frames, interval=f"iv{interval}", band=band, level=0.1, sigma=0.01)
                )
    return spectra


def test_a_constant_footprint_across_levels_needs_only_one() -> None:
    """Same footprint in every band: between-level variation must not exceed the split-half and
    between-interval variation that are present anyway."""
    stability = fp.assess_stability(_corpus({"shadow": 1.5, "mid": 1.5, "high": 1.5}))
    assert stability.one_footprint_suffices is True
    assert "ONE footprint" in stability.summary()


def test_a_level_dependent_footprint_is_flagged() -> None:
    """Footprint genuinely changes with band: between-level variation must exceed the null."""
    stability = fp.assess_stability(_corpus({"shadow": 0.0, "mid": 1.5, "high": 3.0}))
    assert stability.one_footprint_suffices is False
    assert "level-dependent" in stability.summary()


def test_one_band_cannot_answer_the_question() -> None:
    stability = fp.assess_stability(_corpus({"mid": 1.5}))
    assert stability.one_footprint_suffices is None
    assert "cannot assess" in stability.summary()


def test_stability_needs_enough_windows() -> None:
    with pytest.raises(DataError, match="at least 4 window spectra"):
        fp.assess_stability([])


def test_a_window_spectrum_serialises() -> None:
    record = _spectrum(_grain(), interval="a", band="m", level=0.1, sigma=0.01).as_record()
    assert record["interval"] == "a" and record["band"] == "m"
    assert len(record["radial_psd"]) > 4
    assert len(record["horizontal_psd"]) > 4 and len(record["vertical_psd"]) > 4


# ----------------------------------------------------- directional (orientation)


def test_anisotropy_is_no_longer_quantised_to_integers() -> None:
    """Integer half-widths produced the exact 2.0. Interpolated crossings give a continuous
    value, so a 1.6:1 grain reads as ~1.6, not snapped to 1 or 2."""
    spectrum = _spectrum(
        _grain(correlation_px=2.5, aniso=1.6, seed=7), interval="a", band="m", level=0.1, sigma=0.01
    )
    assert spectrum.anisotropy == pytest.approx(1.6, abs=0.5)
    assert spectrum.anisotropy % 1.0 != 0.0, "a continuous width should not land on an integer"


def test_horizontal_and_vertical_profiles_differ_for_anisotropic_grain() -> None:
    aniso = _spectrum(
        _grain(correlation_px=2.5, aniso=2.5, seed=8), interval="a", band="m", level=0.1, sigma=0.01
    )
    same = _spectrum(
        _grain(correlation_px=2.5, aniso=2.5, seed=9), interval="a", band="m", level=0.1, sigma=0.01
    )
    # A profile is identical to itself in every direction.
    for d in fp.DIRECTIONS:
        assert fp.spectral_distance(aniso, aniso, direction=d) == pytest.approx(0.0, abs=1e-9)
    # The horizontal and vertical shapes genuinely differ for anisotropic grain (that is the
    # information radial averaging throws away).
    assert aniso.horizontal_psd != aniso.vertical_psd
    assert aniso.grain_radius_h > aniso.grain_radius_v
    # Two draws of the same footprint stay reasonably close radially (single-window spectra are
    # noisy, especially on the fast-falling vertical axis of a strongly anisotropic grain).
    assert fp.spectral_distance(aniso, same, direction="radial") < 0.6


def test_a_level_dependent_anisotropy_is_flagged_even_when_the_radial_shape_is_stable() -> None:
    """The defect radial averaging cannot see. Same radial correlation radius in every band, but
    the *orientation* changes with level: isotropic in shadow, 3:1 horizontal in the highlights.
    Radial distance may call it stable; the directional profiles must not."""
    stability = fp.assess_stability(
        _corpus(
            {"shadow": 2.0, "mid": 2.0, "high": 2.0},
            aniso_by_band={"shadow": 1.0, "mid": 1.8, "high": 3.0},
        )
    )
    assert stability.one_footprint_suffices is False
    assert stability.direction("horizontal").one_footprint_suffices is False
    assert "LEVEL-DEPENDENT" in stability.summary()


def test_isotropic_constant_footprint_is_stable_in_every_direction() -> None:
    stability = fp.assess_stability(_corpus({"shadow": 1.5, "mid": 1.5, "high": 1.5}))
    assert all(d.one_footprint_suffices for d in stability.directions)


def test_block_peaks_are_measured_on_the_axial_slices() -> None:
    """A codec grid concentrates on the axes; radial averaging dilutes it. Clean grain shows no
    axial peak."""
    spectrum = _spectrum(_grain(correlation_px=1.0), interval="a", band="m", level=0.1, sigma=0.01)
    assert spectrum.block_peak_h < 2.0 and spectrum.block_peak_v < 2.0
