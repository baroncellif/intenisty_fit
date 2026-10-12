import numpy as np
import pandas as pd
import pytest

from intensity_fit.blend import BlendOperator, cluster_frequencies
from intensity_fit.config import FitOptions, load_run_config
from intensity_fit.fitting import fit_mixture
from intensity_fit.lineshape import render_profiles
from intensity_fit.spectrum import read_spectrum
from intensity_fit.tables import build_species_tables
from synth import SpeciesSpec, build_scenario


def _spectrum(centers, amps, fwhm=0.08, lo=99.0, hi=101.5):
    x = np.round(np.arange(lo, hi, 0.01), 2)
    return pd.DataFrame({"freq_MHz": x, "intensity": render_profiles(x, centers, amps, fwhm=fwhm, normalize="peak")})


def test_cluster_frequencies_groups_neighbours_within_tolerance():
    ids = cluster_frequencies(np.array([5.0, 1.0, 1.01, 1.02, 3.0, 3.2]), 0.03)
    assert ids[1] == ids[2] == ids[3]
    assert len({ids[0], ids[1], ids[4], ids[5]}) == 4
    assert ids.max() == 3


def test_cluster_frequencies_zero_tolerance_keeps_every_distinct_line():
    assert cluster_frequencies(np.array([1.0, 2.0, 3.0]), 0.0).tolist() == [0, 1, 2]


def test_exactly_degenerate_lines_are_one_peak_with_the_summed_height():
    spec = _spectrum([100.0, 100.0], [1.0, 2.0])
    op = BlendOperator([100.0, 100.0], spec, "local_max", 0.05, 0.08, "gaussian", 0.03)
    assert op.n_clusters == 1 and op.n_in_cluster.tolist() == [2]
    assert op.exp[0] == pytest.approx(3.0, rel=1e-6)
    assert op.apply(np.array([1.0, 2.0]))[0] == pytest.approx(3.0, rel=1e-6)


def test_partially_overlapping_lines_use_the_peak_of_the_sum_not_the_sum_of_peaks():
    centers, amps = [100.0, 100.06], [1.0, 1.0]
    spec = _spectrum(centers, amps)
    op = BlendOperator(centers, spec, "local_max", 0.10, 0.08, "gaussian", 0.03)
    assert op.n_clusters == 2                      # farther apart than the tolerance: two observed peaks
    model = op.apply(np.array(amps))
    assert model == pytest.approx(op.exp, rel=1e-6)  # model = same extraction applied to the model spectrum
    assert model[0] > 1.0 and model[0] < 2.0       # more than one peak, less than the sum of peak heights


def test_area_extraction_model_is_the_area_inside_the_window():
    spec = _spectrum([100.0], [2.0])  # peak height 2
    op = BlendOperator([100.0], spec, "area", 0.10, 0.08, "gaussian", 0.03)
    sigma = 0.08 / 2.3548
    full_area = 2.0 * sigma * np.sqrt(2 * np.pi)
    # area-normalised line of area A: the model must reproduce the area the extraction sees in the window
    assert op.apply(np.array([full_area]))[0] == pytest.approx(op.exp[0], rel=1e-3)


def test_sample_extraction_model_is_the_value_at_the_nearest_grid_point():
    spec = _spectrum([100.004], [1.5])
    op = BlendOperator([100.004], spec, "sample", 0.05, 0.08, "gaussian", 0.03)
    assert op.apply(np.array([1.5]))[0] == pytest.approx(op.exp[0], rel=1e-6)


def test_lines_outside_the_spectrum_are_invalid():
    spec = _spectrum([100.0], [1.0])
    op = BlendOperator([100.0, 500.0], spec, "local_max", 0.05, 0.08, "gaussian", 0.03)
    assert op.valid.tolist() == [True, False]


def _blended_scenario(tmp_path):
    # both species have transitions on exactly the same frequencies (J = 0..3); species "a" has one extra line (J = 4)
    specs = [SpeciesSpec("a", 3000.0, n_a=5, scale=0.5, T=6.0), SpeciesSpec("b", 3000.0, n_a=4, scale=2.0, T=6.0)]
    return build_scenario(tmp_path, specs)


def _fit(scn, **opts):
    cfg = load_run_config(scn.config_path)
    fo = FitOptions(**{**cfg.raw["fit"], **opts})
    tables = build_species_tables(cfg.species, fo)
    return fit_mixture(tables, read_spectrum(cfg.spectrum_path), fo)


def test_fit_recovers_scales_when_species_share_frequencies(tmp_path):
    res = _fit(_blended_scenario(tmp_path))
    assert res["species"]["a"]["scale"] == pytest.approx(0.5, rel=1e-3)
    assert res["species"]["b"]["scale"] == pytest.approx(2.0, rel=1e-3)
    assert res["species"]["a"]["Trot_K"] == pytest.approx(6.0, rel=1e-3)
    assert res["blend"]["n_peaks"] == 5 and res["blend"]["n_blended_peaks"] == 4 and res["blend"]["n_transitions"] == 9


def test_legacy_one_datum_per_transition_is_biased_for_blends(tmp_path):
    res = _fit(_blended_scenario(tmp_path), blend_tolerance_MHz=0.0)
    assert res["blend"] is None
    assert abs(res["species"]["b"]["scale"] / 2.0 - 1.0) > 0.05


def test_blend_columns_are_added_to_the_observation_table(tmp_path):
    obs = _fit(_blended_scenario(tmp_path))["observation_table"]
    assert {"cluster", "n_in_cluster", "cluster_exp", "cluster_model"} <= set(obs.columns)
    shared = obs[obs.n_in_cluster == 2]
    assert len(shared) == 8
    assert (shared["cluster_model"] - shared["cluster_exp"]).abs().max() < 1e-6


def test_isolated_lines_give_the_same_result_with_and_without_blend(tmp_path):
    scn = build_scenario(tmp_path, [SpeciesSpec("alpha", 3000.0, n_a=5, scale=0.5), SpeciesSpec("beta", 4000.0, n_a=4, scale=2.0)])
    a, b = _fit(scn), _fit(scn, blend_tolerance_MHz=0.0)
    for name in ("alpha", "beta"):
        assert a["species"][name]["scale"] == pytest.approx(b["species"][name]["scale"], rel=1e-5)
    assert a["species"]["alpha"]["Trot_K"] == pytest.approx(b["species"]["alpha"]["Trot_K"], rel=1e-5)


def test_spectrum_mode_blend_is_only_used_for_the_peak_table(tmp_path):
    res = _fit(build_scenario(tmp_path, [SpeciesSpec("alpha", 3000.0, n_a=5, scale=0.5)], normalize="area"),
               fit_mode="spectrum", residual_weighting="none",
               lineshape={"profile": "gaussian", "fwhm_MHz": 0.08, "window_MHz": 0.3, "normalize": "area"})
    assert res["blend"]["diagnostic_only"] is True
    obs = res["observation_table"]
    # model (area units) converted to peak units and extracted like the data: matches the measured peak heights
    assert (obs["cluster_model"] / obs["cluster_exp"] - 1).abs().max() < 1e-3


def test_spectrum_mode_blended_peak_table_sums_degenerate_transitions(tmp_path):
    specs = [SpeciesSpec("a", 3000.0, n_a=5, scale=0.5, T=6.0), SpeciesSpec("b", 3000.0, n_a=4, scale=2.0, T=6.0)]
    res = _fit(build_scenario(tmp_path, specs, normalize="area"), fit_mode="spectrum", residual_weighting="none",
               lineshape={"profile": "gaussian", "fwhm_MHz": 0.08, "window_MHz": 0.3, "normalize": "area"})
    obs = res["observation_table"]
    assert res["blend"]["n_blended_peaks"] == 4
    assert (obs["cluster_model"] / obs["cluster_exp"] - 1).abs().max() < 1e-3
