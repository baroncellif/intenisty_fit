import numpy as np
import pytest

from intensity_fit.config import FitOptions
from intensity_fit.lineshape import lineshape_options, lineshape_values, render_profiles


def _trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def test_gaussian_area_normalisation():
    x = np.linspace(-2, 2, 40001)
    assert _trapz(lineshape_values(x, 0.0, 0.08, "gaussian", "area"), x) == pytest.approx(1.0, rel=1e-6)


def test_lorentzian_area_normalisation():
    x = np.linspace(-500, 500, 2_000_001)
    assert _trapz(lineshape_values(x, 0.0, 0.08, "lorentzian", "area"), x) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("profile", ["gaussian", "lorentzian"])
def test_peak_normalisation_and_fwhm(profile):
    fwhm = 0.2
    peak = lineshape_values(np.array([0.0]), 0.0, fwhm, profile, "peak")
    half = lineshape_values(np.array([fwhm / 2.0, -fwhm / 2.0]), 0.0, fwhm, profile, "peak")
    assert peak[0] == pytest.approx(1.0)
    assert half == pytest.approx([0.5, 0.5])


def test_lineshape_is_centred_on_requested_frequency():
    x = np.linspace(99.0, 101.0, 2001)
    y = lineshape_values(x, 100.3, 0.1, "gaussian", "peak")
    assert x[np.argmax(y)] == pytest.approx(100.3, abs=1e-3)


def test_profile_name_is_case_insensitive():
    a = lineshape_values(np.array([0.1]), 0.0, 0.2, "Gaussian", "peak")
    b = lineshape_values(np.array([0.1]), 0.0, 0.2, "gaussian", "peak")
    assert a == pytest.approx(b)


def test_unknown_profile_and_normalisation_raise():
    with pytest.raises(ValueError, match="profile"):
        lineshape_values([0.0], 0.0, 0.1, "voigt")
    with pytest.raises(ValueError, match="normalization"):
        lineshape_values([0.0], 0.0, 0.1, "gaussian", "height")


def test_zero_fwhm_does_not_divide_by_zero():
    assert np.all(np.isfinite(lineshape_values(np.array([0.0, 1.0]), 0.0, 0.0, "gaussian", "peak")))


def test_render_profiles_is_linear_superposition():
    x = np.linspace(-1, 3, 4001)
    a = render_profiles(x, [0.0], [2.0], 0.1)
    b = render_profiles(x, [2.0], [3.0], 0.1)
    both = render_profiles(x, [0.0, 2.0], [2.0, 3.0], 0.1)
    assert both == pytest.approx(a + b)


def test_render_profiles_area_equals_intensity():
    x = np.linspace(-1, 3, 80001)
    y = render_profiles(x, [0.0, 2.0], [2.0, 3.0], 0.1, normalize="area")
    assert _trapz(y, x) == pytest.approx(5.0, rel=1e-5)


def test_render_profiles_without_lines_is_zero():
    assert render_profiles(np.linspace(0, 1, 5), [], [], 0.1).tolist() == [0.0] * 5


def test_lineshape_options_defaults():
    opts = lineshape_options(FitOptions(extraction_window_MHz=0.07))
    assert opts == {
        "profile": "gaussian", "fwhm_MHz": 0.08, "fit_fwhm": False,
        "fwhm_bounds_MHz": [0.005, 1.0], "window_MHz": 0.07, "normalize": "area",
    }


def test_lineshape_options_overrides():
    fo = FitOptions(lineshape={"profile": "lorentzian", "fwhm_MHz": 0.2, "fit_fwhm": True, "window_MHz": 0.5, "normalize": "peak"})
    opts = lineshape_options(fo)
    assert (opts["profile"], opts["fwhm_MHz"], opts["fit_fwhm"], opts["window_MHz"], opts["normalize"]) == (
        "lorentzian", 0.2, True, 0.5, "peak")
