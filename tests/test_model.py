import numpy as np
import pytest

from intensity_fit.constants import C2, CM1_PER_MHZ
from intensity_fit.model import intensity_at_T_approx, intensity_at_T_exact, intensity_from_lgint


def test_intensity_from_lgint():
    assert intensity_from_lgint([0.0, -1.0, 2.0]).tolist() == pytest.approx([1.0, 0.1, 100.0])
    assert intensity_from_lgint(-3.0) == pytest.approx(1e-3)


def test_approx_is_identity_at_catalogue_temperature():
    out = intensity_at_T_approx([1.0, 2.0], [0.0, 3.0], [3.0, 3.0], 20.0, 20.0)
    assert out.tolist() == pytest.approx([1.0, 2.0])


def test_approx_matches_closed_form():
    I, elo, dr, T, Tc = 0.3, 2.0, 3.0, 5.0, 300.0
    expected = I * (Tc / T) ** (1 + dr / 2) * np.exp(-C2 * elo * (1 / T - 1 / Tc))
    assert intensity_at_T_approx(I, elo, dr, T, Tc) == pytest.approx(expected)


def test_approx_cooling_favours_low_energy_lines():
    lo = intensity_at_T_approx(1.0, 0.0, 3.0, 5.0, 300.0)
    hi = intensity_at_T_approx(1.0, 10.0, 3.0, 5.0, 300.0)
    assert hi < lo


def test_approx_accepts_per_line_catalogue_temperature():
    out = intensity_at_T_approx([1.0, 1.0], [0.0, 0.0], [2.0, 2.0], 10.0, np.array([10.0, 20.0]))
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(2.0 ** 2)


def test_approx_returns_arrays_for_vector_input():
    out = intensity_at_T_approx(np.ones(4), np.arange(4.0), np.full(4, 3.0), 5.0, 300.0)
    assert out.shape == (4,)


def test_exact_is_identity_at_catalogue_temperature():
    q = lambda T: T ** 1.5
    out = intensity_at_T_exact(np.array([1.0, 2.0]), np.array([1.0, 4.0]), np.array([5000.0, 9000.0]), 20.0, 20.0, q)
    assert out.tolist() == pytest.approx([1.0, 2.0])


def test_exact_matches_closed_form():
    q = lambda T: T ** 1.5
    I, elo, nu, T, Tc = 0.7, 2.0, 8000.0, 5.0, 300.0
    eup = elo + nu * CM1_PER_MHZ
    num = np.exp(-C2 * elo / T) - np.exp(-C2 * eup / T)
    den = np.exp(-C2 * elo / Tc) - np.exp(-C2 * eup / Tc)
    expected = I * q(Tc) / q(T) * num / den
    out = intensity_at_T_exact(np.array([I]), np.array([elo]), np.array([nu]), T, Tc, q)
    assert out[0] == pytest.approx(expected)


def test_exact_and_approx_agree_for_low_frequency_lines():
    """For h*nu << kT the two scalings coincide when Q ~ T^(DR/2)."""
    q = lambda T: T ** 1.5
    elo = np.array([0.0, 1.0, 5.0])
    nu = np.full(3, 50.0)
    exact = intensity_at_T_exact(np.ones(3), elo, nu, 20.0, 300.0, q)
    approx = intensity_at_T_approx(np.ones(3), elo, np.full(3, 3.0), 20.0, 300.0)
    assert exact == pytest.approx(approx, rel=1e-2)
