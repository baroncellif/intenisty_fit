import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pytest

from intensity_fit import plotting, workflow
from synth import SpeciesSpec, build_scenario, true_intensity, transitions


@pytest.fixture
def fitted(tmp_path):
    specs = [SpeciesSpec("alpha", 3000.0, n_a=4, n_c=2, scale=0.5), SpeciesSpec("beta", 4000.0, n_a=4, scale=2.0)]
    scn = build_scenario(tmp_path, specs)
    out = workflow.run_from_config(scn.config_path)
    return out


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _opts(out, **plot):
    return dataclasses.replace(out["config"].fit_options, plot=plot)


def test_gaussian_broadening_peak_height_and_position():
    x = np.linspace(99.0, 101.0, 2001)
    y = plotting.gaussian_broadening(x, [100.0], [3.0], fwhm_MHz=0.1)
    assert y.max() == pytest.approx(3.0, rel=1e-3)
    assert x[np.argmax(y)] == pytest.approx(100.0, abs=1e-3)


def test_prediction_table_for_plot_matches_truth(fitted):
    pred = plotting.build_prediction_table_for_plot(fitted["species_tables"], fitted["result"], fitted["config"].fit_options)
    specs = [SpeciesSpec("alpha", 3000.0, n_a=4, n_c=2, scale=0.5), SpeciesSpec("beta", 4000.0, n_a=4, scale=2.0)]
    expected = [true_intensity(t, s) for s in specs for t in transitions(s)]
    assert pred["pred_intensity"].tolist() == pytest.approx(expected, rel=1e-3)
    assert set(pred["species"]) == {"alpha", "beta"}


def test_prediction_table_can_use_catalogue_frequencies(fitted):
    pred = plotting.build_prediction_table_for_plot(
        fitted["species_tables"], fitted["result"], fitted["config"].fit_options, use_lin_frequency=False)
    assert pred["freq_MHz"].tolist() == pytest.approx(
        np.concatenate([t["cat_freq_MHz"].to_numpy() for t in fitted["species_tables"].values()]).tolist())


def test_result_fwhm_prefers_fitted_value(fitted):
    opts = fitted["config"].fit_options
    assert plotting.result_fwhm(None, opts) == pytest.approx(0.08)
    assert plotting.result_fwhm({"fit_stats": {"lineshape_fwhm_MHz": 0.2}}, opts) == pytest.approx(0.2)


def test_overlay_arrays_model_matches_experiment_shape(fitted):
    x, y_exp, y_fit, fmin, fmax = plotting._overlay_arrays(
        fitted["spectrum_df"], fitted["line_diag_df"], _opts(fitted), result=fitted["result"])
    assert x.size == y_exp.size == y_fit.size > 0
    assert y_exp.max() == pytest.approx(1.0) and y_fit.max() == pytest.approx(1.0)
    assert fmin < 3000.0 and fmax > 4300.0


def test_overlay_arrays_without_normalisation_reproduces_model(fitted):
    x, y_exp, y_fit, *_ = plotting._overlay_arrays(
        fitted["spectrum_df"], fitted["line_diag_df"], dataclasses.replace(
            fitted["config"].fit_options, lineshape={"normalize": "peak", "fwhm_MHz": 0.08},
            plot={"normalize_exp": False, "normalize_pred": False}),
        result={"fit_stats": {"lineshape_fwhm_MHz": 0.08}})
    assert y_fit == pytest.approx(y_exp, rel=1e-3, abs=1e-6)


def test_overlay_arrays_empty_diagnostics(fitted):
    x, *_ = plotting._overlay_arrays(fitted["spectrum_df"], fitted["line_diag_df"].iloc[0:0], _opts(fitted))
    assert x.size == 0


def test_quick_overlay_png(fitted, tmp_path):
    path = tmp_path / "o.png"
    plotting.write_quick_overlay_plot(path, fitted["spectrum_df"], fitted["line_diag_df"], _opts(fitted), result=fitted["result"])
    assert path.stat().st_size > 1000


def test_quick_overlay_png_empty_does_nothing(fitted, tmp_path):
    path = tmp_path / "o.png"
    plotting.write_quick_overlay_plot(path, fitted["spectrum_df"], fitted["line_diag_df"].iloc[0:0], _opts(fitted))
    assert not path.exists()


def test_species_pdf_paginates(fitted, tmp_path):
    path = tmp_path / "o.pdf"
    plotting.write_species_line_pdf(path, fitted["spectrum_df"], fitted["line_diag_df"],
                                    _opts(fitted, lines_per_page=5, ncols=2), result=fitted["result"])
    assert path.read_bytes().count(b"/Type /Page\n") + path.read_bytes().count(b"/Type /Page ") >= 2


def test_species_pdf_empty_does_nothing(fitted, tmp_path):
    path = tmp_path / "o.pdf"
    plotting.write_species_line_pdf(path, fitted["spectrum_df"], fitted["line_diag_df"].iloc[0:0], _opts(fitted))
    assert not path.exists()


@pytest.mark.parametrize("mode", ["sticks", "broadened"])
@pytest.mark.parametrize("color_by", ["species", "component", "species_component"])
def test_interactive_plot_builds_a_figure(fitted, mode, color_by):
    plotting.show_interactive_overlay_plot(
        fitted["spectrum_df"], fitted["line_diag_df"],
        _opts(fitted, interactive_mode=mode, interactive_color_by=color_by, interactive_npoints=2000, show_total=True),
        result=fitted["result"])
    assert plt.get_fignums()


def test_interactive_plot_rejects_bad_options(fitted):
    with pytest.raises(ValueError, match="interactive_color_by"):
        plotting.show_interactive_overlay_plot(
            fitted["spectrum_df"], fitted["line_diag_df"], _opts(fitted, interactive_color_by="bogus"), result=fitted["result"])
    with pytest.raises(ValueError, match="interactive_mode"):
        plotting.show_interactive_overlay_plot(
            fitted["spectrum_df"], fitted["line_diag_df"], _opts(fitted, interactive_mode="bogus", interactive_npoints=500),
            result=fitted["result"])
