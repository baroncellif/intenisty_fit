import numpy as np
import pandas as pd
import pytest

from intensity_fit.spectrum import extract_line_intensity, integrate_area, read_spectrum


def test_read_spectrum_two_columns(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("1.0 10\n2.0 20\n")
    df = read_spectrum(str(path))
    assert df.columns.tolist() == ["freq_MHz", "intensity", "sigma"]
    assert df["sigma"].isna().all()


def test_read_spectrum_three_columns_and_comments(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("# freq intensity sigma\n1.0 10 0.5\n2.0 20 0.6\n")
    df = read_spectrum(str(path))
    assert df["sigma"].tolist() == [0.5, 0.6]
    assert len(df) == 2


def test_read_spectrum_sorts_by_frequency(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("3 30\n1 10\n2 20\n")
    assert read_spectrum(str(path))["freq_MHz"].tolist() == [1.0, 2.0, 3.0]


def test_read_spectrum_needs_two_columns(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("1\n2\n")
    with pytest.raises(ValueError, match="at least 2 columns"):
        read_spectrum(str(path))


def test_integrate_area_trapezoid():
    x = np.linspace(0, 2, 201)
    assert integrate_area(x, x) == pytest.approx(2.0)


@pytest.fixture
def spec():
    x = np.round(np.arange(99.0, 101.0001, 0.01), 2)
    y = np.exp(-0.5 * ((x - 100.0) / 0.05) ** 2)
    return pd.DataFrame({"freq_MHz": x, "intensity": y})


def test_extract_local_max(spec):
    assert extract_line_intensity(spec, np.array([100.0]), "local_max", 0.05)[0] == pytest.approx(1.0)


def test_extract_sample_picks_nearest_point(spec):
    out = extract_line_intensity(spec, np.array([100.004]), "sample", 0.05)
    assert out[0] == pytest.approx(1.0)
    off = extract_line_intensity(spec, np.array([100.04]), "sample", 0.05)
    assert off[0] == pytest.approx(np.exp(-0.5 * (0.04 / 0.05) ** 2))


def test_extract_area_of_constant_signal():
    x = np.round(np.arange(0.0, 10.0001, 0.01), 2)
    df = pd.DataFrame({"freq_MHz": x, "intensity": np.full_like(x, 2.0)})
    out = extract_line_intensity(df, np.array([5.0]), "area", 0.05)
    assert out[0] == pytest.approx(2.0 * 0.10, rel=1e-6)


def test_extract_area_of_gaussian_matches_analytic(spec):
    out = extract_line_intensity(spec, np.array([100.0]), "area", 0.5)
    assert out[0] == pytest.approx(0.05 * np.sqrt(2 * np.pi), rel=1e-4)


def test_extract_outside_spectrum_is_nan(spec):
    out = extract_line_intensity(spec, np.array([100.0, 500.0]), "local_max", 0.05)
    assert np.isfinite(out[0])
    assert np.isnan(out[1])


def test_extract_unknown_mode_raises(spec):
    with pytest.raises(ValueError, match="extraction mode"):
        extract_line_intensity(spec, np.array([100.0]), "median", 0.05)


def test_extract_window_edges_are_inclusive():
    df = pd.DataFrame({"freq_MHz": [9.95, 10.05], "intensity": [1.0, 3.0]})
    assert extract_line_intensity(df, np.array([10.0]), "local_max", 0.05)[0] == 3.0
