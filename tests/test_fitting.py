import numpy as np
import pytest

from intensity_fit import fitting
from intensity_fit.config import FitOptions, load_run_config
from intensity_fit.fitting import fit_mixture
from intensity_fit.pickett_meta import make_partition_function
from intensity_fit.spectrum import read_spectrum
from intensity_fit.tables import build_species_tables
from synth import SpeciesSpec, build_scenario


def run_fit(scn, **overrides):
    cfg = load_run_config(scn.config_path)
    opts = FitOptions(**{**cfg.raw["fit"], **overrides})
    tables = build_species_tables(cfg.species, opts)
    q = {n: make_partition_function(t.attrs["partition_table"]) for n, t in tables.items()}
    q = {n: f for n, f in q.items() if f is not None}
    result = fit_mixture(tables, read_spectrum(cfg.spectrum_path), opts, q_functions=q)
    return result, tables, opts


# ---------- weighting ----------

def test_sigma_none_is_unity():
    y = np.array([1.0, 5.0])
    assert fitting._intensity_sigma(y, FitOptions(), "none").tolist() == [1.0, 1.0]


def test_sigma_fractional_scales_with_intensity():
    s = fitting._intensity_sigma(np.array([1.0, -5.0, 0.0]), FitOptions(intensity_uncertainty_fraction=0.1), "fractional")
    assert s[:2].tolist() == pytest.approx([0.1, 0.5])
    assert s[2] == np.finfo(float).tiny


def test_sigma_hybrid_adds_floor_in_quadrature():
    fo = FitOptions(intensity_uncertainty_fraction=0.1, intensity_uncertainty_floor=0.3)
    s = fitting._intensity_sigma(np.array([4.0, 0.0]), fo, "hybrid")
    assert s.tolist() == pytest.approx([np.hypot(0.4, 0.3), 0.3])


def test_sigma_hybrid_without_floor_equals_fractional():
    fo = FitOptions(intensity_uncertainty_fraction=0.2)
    y = np.array([1.0, 2.0])
    assert fitting._intensity_sigma(y, fo, "hybrid").tolist() == pytest.approx(fitting._intensity_sigma(y, fo, "fractional").tolist())


def test_sigma_rejects_negative_floor_and_unknown_mode():
    with pytest.raises(ValueError, match="floor"):
        fitting._intensity_sigma(np.ones(2), FitOptions(intensity_uncertainty_floor=-1.0), "hybrid")
    with pytest.raises(ValueError, match="residual_weighting"):
        fitting._intensity_sigma(np.ones(2), FitOptions(), "bogus")


# ---------- temperature groups ----------

NAMES = ["a", "b", "c"]


def test_temperature_structure_common():
    ts = fitting.build_group_temperature_structure(NAMES, FitOptions())
    assert ts["group_names"] == ["ALL"]
    assert ts["species_to_group"] == {n: "ALL" for n in NAMES}
    assert ts["fitted_group_names"] == ["ALL"]


def test_temperature_structure_one_per_species():
    ts = fitting.build_group_temperature_structure(NAMES, FitOptions(common_temperature=False))
    assert ts["group_names"] == NAMES
    assert ts["species_to_group"] == {n: n for n in NAMES}


def test_temperature_structure_explicit_groups_with_fixed_group():
    fo = FitOptions(species_temperature_groups={"a": "g1", "b": "g1", "c": "g2"}, fixed_group_temperatures={"g2": 3.0})
    ts = fitting.build_group_temperature_structure(NAMES, fo)
    assert ts["group_names"] == ["g1", "g2"]
    assert ts["fitted_group_names"] == ["g1"]
    assert ts["fixed_group_temperatures"] == {"g2": 3.0}


def test_temperature_structure_fixed_global():
    ts = fitting.build_group_temperature_structure(NAMES, FitOptions(fixed_temperature_K=4.0))
    assert ts["mode"] == "fixed_global"
    assert ts["fitted_group_names"] == []


def test_temperature_structure_errors():
    with pytest.raises(ValueError, match="Missing temperature group"):
        fitting.build_group_temperature_structure(NAMES, FitOptions(species_temperature_groups={"a": "g"}))
    with pytest.raises(ValueError, match="unknown groups"):
        fitting.build_group_temperature_structure(NAMES, FitOptions(fixed_group_temperatures={"zzz": 1.0}))


def test_evaluate_species_temperatures_grouped():
    fo = FitOptions(species_temperature_groups={"a": "g1", "b": "g1", "c": "g2"}, fixed_group_temperatures={"g2": 3.0})
    ts = fitting.build_group_temperature_structure(NAMES, fo)
    p = np.r_[np.log(2.0) * np.ones(3), np.log(7.0)]  # 3 scales, then ln T(g1)
    per_species, per_group = fitting.evaluate_species_temperatures(NAMES, fo, ts, p, 3)
    assert per_species == pytest.approx({"a": 7.0, "b": 7.0, "c": 3.0})
    assert per_group == pytest.approx({"g1": 7.0, "g2": 3.0})


def test_evaluate_species_temperatures_fixed_global():
    fo = FitOptions(fixed_temperature_K=4.0)
    ts = fitting.build_group_temperature_structure(NAMES, fo)
    per_species, per_group = fitting.evaluate_species_temperatures(NAMES, fo, ts, np.zeros(3), 3)
    assert per_species == {n: 4.0 for n in NAMES}
    assert per_group == {"GLOBAL_FIXED": 4.0}


# ---------- full fits on synthetic data ----------

def test_fit_recovers_scales_and_temperature(two_species):
    res, *_ = run_fit(two_species)
    assert res["success"]
    for spec in two_species.specs:
        sp = res["species"][spec.name]
        assert sp["scale"] == pytest.approx(spec.scale, rel=1e-4)
        assert sp["Trot_K"] == pytest.approx(spec.T, rel=1e-4)
    assert res["fit_stats"]["raw_rmse"] < 1e-5
    assert res["fit_stats"]["n_obs"] == 9
    assert res["fit_stats"]["n_par"] == 3  # 2 scales + 1 common T


def test_fit_relative_abundances(two_species):
    res, *_ = run_fit(two_species)
    rel = {n: v["relative_abundance"] for n, v in res["species"].items()}
    assert sum(rel.values()) == pytest.approx(1.0)
    assert rel["alpha"] == pytest.approx(0.5 / 2.5, rel=1e-4)
    assert res["species"]["beta"]["population_relmax"] == pytest.approx(1.0)
    assert res["species"]["alpha"]["population_relmax"] == pytest.approx(0.25, rel=1e-4)


def test_fit_reports_uncertainties_and_monte_carlo(two_species):
    res, *_ = run_fit(two_species)
    assert res["temperature_groups"]["ALL"]["Trot_K"] == pytest.approx(6.0, rel=1e-4)
    assert res["temperature_groups"]["ALL"]["Trot_err_K"] >= 0.0
    assert res["species"]["alpha"]["scale_err"] >= 0.0
    assert res["fit_stats"]["n_mc"] == 200
    assert res["species"]["alpha"]["relative_abundance_mc_rms"] >= 0.0
    cov = np.array(res["covariance_matrix"])
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)


def test_fit_is_reproducible_with_seed(two_species):
    a, *_ = run_fit(two_species)
    b, *_ = run_fit(two_species)
    assert a["species"]["alpha"]["scale_mc_rms"] == b["species"]["alpha"]["scale_mc_rms"]


def test_fit_with_monte_carlo_disabled(two_species):
    res, *_ = run_fit(two_species, n_mc=0)
    assert res["fit_stats"]["n_mc"] == 0
    assert "scale_mc_rms" not in res["species"]["alpha"]


def test_fit_with_fixed_temperature(two_species):
    res, *_ = run_fit(two_species, fixed_temperature_K=6.0)
    assert res["Trot_fixed_K"] == 6.0
    assert res["species"]["alpha"]["temperature_group"] == "GLOBAL_FIXED"
    assert res["species"]["alpha"]["scale"] == pytest.approx(0.5, rel=1e-5)
    assert res["fit_stats"]["n_par"] == 2


def test_fit_with_wrong_fixed_temperature_degrades_quality(two_species):
    good, *_ = run_fit(two_species, fixed_temperature_K=6.0)
    bad, *_ = run_fit(two_species, fixed_temperature_K=12.0)
    assert bad["fit_stats"]["rmse"] > 100 * good["fit_stats"]["rmse"] + 1e-6


def test_fit_with_separate_temperatures(tmp_path):
    specs = [SpeciesSpec("alpha", 3000.0, n_a=6, scale=0.5, T=4.0), SpeciesSpec("beta", 4000.0, n_a=6, scale=2.0, T=9.0)]
    res, *_ = run_fit(build_scenario(tmp_path, specs), common_temperature=False)
    assert res["species"]["alpha"]["Trot_K"] == pytest.approx(4.0, rel=1e-4)
    assert res["species"]["beta"]["Trot_K"] == pytest.approx(9.0, rel=1e-4)
    assert res["species"]["alpha"]["scale"] == pytest.approx(0.5, rel=1e-4)
    assert res["species"]["beta"]["scale"] == pytest.approx(2.0, rel=1e-4)


def test_fit_with_temperature_groups_and_fixed_group(tmp_path):
    specs = [SpeciesSpec("a", 3000.0, n_a=6, scale=1.0, T=5.0), SpeciesSpec("b", 4000.0, n_a=6, scale=3.0, T=5.0),
             SpeciesSpec("c", 5000.0, n_a=6, scale=0.7, T=8.0)]
    res, *_ = run_fit(build_scenario(tmp_path, specs), species_temperature_groups={"a": "g1", "b": "g1", "c": "g2"},
                      fixed_group_temperatures={"g2": 8.0})
    assert res["temperature_groups"]["g2"]["is_fixed"] is True
    assert np.isnan(res["temperature_groups"]["g2"]["Trot_err_K"])
    assert res["temperature_groups"]["g1"]["Trot_K"] == pytest.approx(5.0, rel=1e-4)
    assert res["species"]["c"]["scale"] == pytest.approx(0.7, rel=1e-4)


def test_fixed_scale_species_is_not_fitted(two_species):
    cfg = load_run_config(two_species.config_path)
    opts = FitOptions(**cfg.raw["fit"])
    cfg.species[0].fixed_scale = 0.5
    tables = build_species_tables(cfg.species, opts)
    res = fit_mixture(tables, read_spectrum(cfg.spectrum_path), opts)
    assert res["species"]["alpha"]["scale_fixed"] is True
    assert res["species"]["alpha"]["scale"] == 0.5
    assert res["species"]["beta"]["scale"] == pytest.approx(2.0, rel=1e-4)
    assert res["fit_stats"]["n_par"] == 2  # beta scale + T
    assert res["fit_stats"]["n_mc"] == 0  # MC skipped when any scale is fixed


def test_fit_in_log_space_recovers_truth(two_species):
    res, *_ = run_fit(two_species, fit_in_log_space=True)
    assert res["species"]["beta"]["scale"] == pytest.approx(2.0, rel=1e-3)
    assert res["species"]["beta"]["Trot_K"] == pytest.approx(6.0, rel=1e-3)


@pytest.mark.parametrize("weighting", ["none", "fractional", "hybrid"])
def test_fit_works_for_every_weighting(two_species, weighting):
    res, *_ = run_fit(two_species, residual_weighting=weighting, intensity_uncertainty_floor=0.01)
    assert res["species"]["alpha"]["scale"] == pytest.approx(0.5, rel=1e-3)


@pytest.mark.parametrize("loss", ["soft_l1", "huber", "cauchy"])
def test_fit_with_robust_losses(two_species, loss):
    res, *_ = run_fit(two_species, loss=loss)
    assert res["species"]["beta"]["Trot_K"] == pytest.approx(6.0, rel=1e-3)


def test_robust_loss_resists_a_corrupted_line(two_species):
    cfg = load_run_config(two_species.config_path)
    spec = read_spectrum(cfg.spectrum_path)
    centre = 3200.0  # an alpha line
    spec.loc[(spec["freq_MHz"] - centre).abs() < 0.05, "intensity"] *= 20.0
    opts = FitOptions(**cfg.raw["fit"])
    tables = build_species_tables(cfg.species, opts)
    plain = fit_mixture(tables, spec, opts)
    robust = fit_mixture(tables, spec, FitOptions(**{**cfg.raw["fit"], "loss": "cauchy"}))
    err = lambda r: abs(r["species"]["alpha"]["scale"] - 0.5)
    assert err(robust) < err(plain)


def test_exact_temperature_scaling_close_to_approx(two_species):
    res_approx, *_ = run_fit(two_species)
    res_exact, *_ = run_fit(two_species, use_exact_temperature_scaling=True)
    assert res_exact["success"]
    assert res_exact["species"]["alpha"]["Trot_K"] == pytest.approx(res_approx["species"]["alpha"]["Trot_K"], rel=0.05)


def test_spectrum_mode_recovers_truth(tmp_path):
    specs = [SpeciesSpec("alpha", 3000.0, n_a=5, scale=0.5, T=6.0), SpeciesSpec("beta", 4000.0, n_a=4, scale=2.0, T=6.0)]
    scn = build_scenario(tmp_path, specs, normalize="area")
    res, *_ = run_fit(scn, fit_mode="spectrum", residual_weighting="none",
                      lineshape={"profile": "gaussian", "fwhm_MHz": 0.08, "window_MHz": 0.3, "normalize": "area"})
    assert res["species"]["alpha"]["scale"] == pytest.approx(0.5, rel=1e-3)
    assert res["species"]["beta"]["Trot_K"] == pytest.approx(6.0, rel=1e-3)
    assert res["fit_stats"]["lineshape_fwhm_fitted"] is False
    assert res["fit_stats"]["fit_mode"] == "spectrum"


def test_spectrum_mode_can_fit_fwhm(tmp_path):
    specs = [SpeciesSpec("alpha", 3000.0, n_a=5, scale=0.5, T=6.0), SpeciesSpec("beta", 4000.0, n_a=4, scale=2.0, T=6.0)]
    scn = build_scenario(tmp_path, specs, normalize="area")
    res, *_ = run_fit(scn, fit_mode="spectrum", residual_weighting="none",
                      lineshape={"profile": "gaussian", "fwhm_MHz": 0.12, "fit_fwhm": True, "window_MHz": 0.3, "normalize": "area"})
    assert res["fit_stats"]["lineshape_fwhm_fitted"] is True
    assert res["fit_stats"]["lineshape_fwhm_MHz"] == pytest.approx(0.08, rel=1e-2)


def test_spectrum_mode_with_no_points_in_windows_raises(two_species):
    with pytest.raises(ValueError, match="selected no spectrum points"):
        fitting._spectrum_points_for_windows(read_spectrum(two_species.spectrum_path), np.array([1.0]), 0.1)


def test_spectrum_points_for_windows_selects_union():
    import pandas as pd
    df = pd.DataFrame({"freq_MHz": [1.0, 2.0, 3.0, 10.0], "intensity": [1, 2, 3, 4.0]})
    x, y = fitting._spectrum_points_for_windows(df, np.array([1.5, 10.0]), 0.6)
    assert x.tolist() == [1.0, 2.0, 10.0]
    assert y.tolist() == [1.0, 2.0, 4.0]


# ---------- dipole targets ----------

def _tables_with_components(tmp_path, **cfg_overrides):
    scn = build_scenario(tmp_path, [SpeciesSpec("mix", 3000.0, n_a=4, n_c=4, scale=1.0, T=6.0)])
    cfg = load_run_config(scn.config_path)
    for k, v in cfg_overrides.items():
        setattr(cfg.species[0], k, v)
    opts = FitOptions(**cfg.raw["fit"])
    return build_species_tables(cfg.species, opts), opts


def test_dipole_targets_from_species_and_global(tmp_path):
    tables, opts = _tables_with_components(tmp_path, fit_dipole_components=["a", "C", "a"])
    assert fitting._build_dipole_targets(tables, opts) == [("mix", "a"), ("mix", "c")]
    tables, opts = _tables_with_components(tmp_path)
    opts.fit_dipole_components = ["b"]
    assert fitting._build_dipole_targets(tables, opts) == [("*", "b")]


def test_dipole_targets_reject_unknown_component(tmp_path):
    tables, opts = _tables_with_components(tmp_path, fit_dipole_components=["x"])
    with pytest.raises(ValueError, match="'a', 'b', or 'c'"):
        fitting._build_dipole_targets(tables, opts)
    tables, opts = _tables_with_components(tmp_path)
    opts.fit_dipole_components = ["q"]
    with pytest.raises(ValueError):
        fitting._build_dipole_targets(tables, opts)


def test_dipole_target_keys_and_formatting():
    assert fitting._dipole_target_key(("sp", "a")) == "sp:mu_a"
    assert fitting._dipole_target_key(("*", "c")) == "GLOBAL:mu_c"
    formatted = fitting._format_dipole_scales({("sp", "a"): 1.2})
    assert formatted["sp:mu_a"]["scale"] == 1.2
    assert np.isnan(formatted["sp:mu_a"]["scale_err"])


def test_dipole_target_line_mask(tmp_path):
    scn = build_scenario(tmp_path, [SpeciesSpec("mix", 3000.0, n_a=4, n_c=4)])
    cfg = load_run_config(scn.config_path)
    opts = FitOptions(**cfg.raw["fit"])
    tables = build_species_tables(cfg.species, opts)
    from intensity_fit.tables import build_observation_table
    obs = build_observation_table(tables, read_spectrum(cfg.spectrum_path), opts)
    assert fitting._dipole_target_line_mask(obs, [("mix", "c")]).sum() == 4
    assert fitting._dipole_target_line_mask(obs, [("*", "a")]).sum() == 4
    assert fitting._dipole_target_line_mask(obs, [("other", "a")]).sum() == 0


# ---------- dipole fits ----------

@pytest.fixture
def dipole_scn(tmp_path):
    """alpha fixes T and fixes nothing else; beta (fixed scale) has its c-lines scaled by mu_scale_c**2."""
    specs = [
        SpeciesSpec("alpha", 3000.0, n_a=5, scale=0.8, T=6.0),
        SpeciesSpec("beta", 4000.0, n_a=4, n_c=4, scale=1.0, T=6.0, mu_scale_c=1.3),
    ]
    return build_scenario(tmp_path, specs)


def _beta_dipole_cfg(scn, **fit):
    cfg = load_run_config(scn.config_path)
    cfg.species[1].fixed_scale = 1.0
    cfg.species[1].fit_dipole_components = ["c"]
    opts = FitOptions(**{**cfg.raw["fit"], "fixed_temperature_K": 6.0, "dipole_model": "component_scaled", **fit})
    return build_species_tables(cfg.species, opts), read_spectrum(cfg.spectrum_path), opts


def test_single_step_dipole_fit_recovers_mu_scale(dipole_scn):
    tables, spectrum, opts = _beta_dipole_cfg(dipole_scn)
    res = fit_mixture(tables, spectrum, opts)
    assert res["dipole_scales"]["beta:mu_c"]["scale"] == pytest.approx(1.3, rel=1e-4)
    assert res["species"]["alpha"]["scale"] == pytest.approx(0.8, rel=1e-4)
    assert any("correlated" in w for w in res["warnings"])


def test_second_step_dipole_fit_recovers_mu_scale(dipole_scn):
    tables, spectrum, opts = _beta_dipole_cfg(dipole_scn, dipole_fit_second_step=True)
    res = fit_mixture(tables, spectrum, opts)
    assert res["dipole_scales"]["beta:mu_c"]["scale"] == pytest.approx(1.3, rel=1e-4)
    diag = res["dipole_group_diagnostics"][0]
    assert (diag["species"], diag["component"], diag["status"]) == ("beta", "c", "ok")
    assert diag["n_used"] == diag["n_total"] == 4
    assert res["fit_stats"]["dipole_fit_second_step"] is True
    assert res["species"]["alpha"]["scale"] == pytest.approx(0.8, rel=1e-4)  # first-step values carried over
    assert res["first_step_result"]["fit_stats"]["n_par"] == 1
    assert res["dipole_rejected_lines"] == []
    assert any("Second-step" in w for w in res["warnings"])
    assert not any("large conditional mu scale" in w for w in res["warnings"])


def test_second_step_flags_large_mu_scale(tmp_path):
    specs = [SpeciesSpec("alpha", 3000.0, n_a=5, scale=0.8, T=6.0),
             SpeciesSpec("beta", 4000.0, n_a=4, n_c=4, scale=1.0, T=6.0, mu_scale_c=2.0)]
    tables, spectrum, opts = _beta_dipole_cfg(build_scenario(tmp_path, specs), dipole_fit_second_step=True)
    res = fit_mixture(tables, spectrum, opts)
    assert res["dipole_scales"]["beta:mu_c"]["scale"] == pytest.approx(2.0, rel=1e-4)
    assert any("large conditional mu scale" in w for w in res["warnings"])


def test_second_step_rejects_outlier_line(dipole_scn):
    tables, spectrum, opts = _beta_dipole_cfg(dipole_scn, dipole_fit_second_step=True, dipole_fit_reject_sigma=3.0)
    c_line = tables["beta"].query("dipole_component == 'c'")["lin_freq_MHz"].iloc[1]
    spectrum.loc[(spectrum["freq_MHz"] - c_line).abs() < 0.05, "intensity"] *= 30.0
    res = fit_mixture(tables, spectrum, opts)
    diag = res["dipole_group_diagnostics"][0]
    assert diag["n_rejected"] == 1
    assert diag["n_used"] == 3
    assert res["dipole_scales"]["beta:mu_c"]["scale"] == pytest.approx(1.3, rel=1e-2)
    assert [r["species"] for r in res["dipole_rejected_lines"]] == ["beta"]
    assert res["dipole_rejected_lines"][0]["reject_reason"] == "large_log_residual_after_component_prelim_fit"


def test_second_step_without_outlier_rejection_keeps_all_lines(dipole_scn):
    tables, spectrum, opts = _beta_dipole_cfg(dipole_scn, dipole_fit_second_step=True, dipole_fit_reject_outliers=False)
    c_line = tables["beta"].query("dipole_component == 'c'")["lin_freq_MHz"].iloc[1]
    spectrum.loc[(spectrum["freq_MHz"] - c_line).abs() < 0.05, "intensity"] *= 30.0
    diag = fit_mixture(tables, spectrum, opts)["dipole_group_diagnostics"][0]
    assert diag["n_used"] == 4


def test_second_step_with_too_few_lines_leaves_scale_at_one(dipole_scn):
    tables, spectrum, opts = _beta_dipole_cfg(dipole_scn, dipole_fit_second_step=True, dipole_fit_min_lines=10)
    res = fit_mixture(tables, spectrum, opts)
    assert res["dipole_scales"]["beta:mu_c"]["scale"] == 1.0
    assert res["dipole_group_diagnostics"][0]["status"] == "insufficient_accepted_lines"
    assert res["success"] is False
    assert any("left at 1.0" in w for w in res["warnings"])


def test_dipole_model_from_cat_ignores_requested_components(dipole_scn):
    tables, spectrum, opts = _beta_dipole_cfg(dipole_scn, dipole_model="from_cat")
    res = fit_mixture(tables, spectrum, opts)
    assert res["dipole_scales"] == {}


def test_warning_when_requested_component_has_no_lines(tmp_path):
    scn = build_scenario(tmp_path, [SpeciesSpec("aonly", 3000.0, n_a=5, scale=1.0, T=6.0)])
    cfg = load_run_config(scn.config_path)
    cfg.species[0].fit_dipole_components = ["c"]
    opts = FitOptions(**cfg.raw["fit"])
    tables = build_species_tables(cfg.species, opts)
    warnings = fitting._build_fit_warnings({}, tables, [("aonly", "c")])
    assert any("no matched c-type fitted lines" in w for w in warnings)


def test_warning_when_int_has_no_requested_dipole(tmp_path):
    scn = build_scenario(tmp_path, [SpeciesSpec("noint", 3000.0, n_a=3, n_c=3, with_int=False)])
    cfg = load_run_config(scn.config_path)
    opts = FitOptions(**cfg.raw["fit"])
    tables = build_species_tables(cfg.species, opts)
    warnings = fitting._build_fit_warnings({}, tables, [("noint", "c")])
    assert any("no mu_c value was read from the .int" in w for w in warnings)


def test_no_dipole_targets_no_warnings():
    assert fitting._build_fit_warnings({}, {}, []) == []


# ---------- correlated spectrum points ----------

def test_autocorr_factor_of_white_noise_is_one():
    from intensity_fit.fitting import residual_autocorr_factor
    rng = np.random.default_rng(0)
    x = np.arange(20000) * 0.01
    assert residual_autocorr_factor(x, rng.normal(size=x.size)) == pytest.approx(1.0, abs=0.15)


def test_autocorr_factor_of_ar1_noise():
    from intensity_fit.fitting import residual_autocorr_factor
    rng = np.random.default_rng(1)
    n, rho = 60000, 0.9
    e = rng.normal(size=n); r = np.zeros(n)
    for i in range(1, n):
        r[i] = rho * r[i - 1] + e[i]
    x = np.arange(n) * 0.01
    assert residual_autocorr_factor(x, r) == pytest.approx((1 + rho) / (1 - rho), rel=0.2)   # = 19 for rho = 0.9


def test_autocorr_factor_does_not_pair_points_across_gaps():
    from intensity_fit.fitting import residual_autocorr_factor
    rng = np.random.default_rng(2)
    n = 4000
    x = np.concatenate([np.arange(n) * 0.01, 1000.0 + np.arange(n) * 0.01])
    r = np.concatenate([rng.normal(size=n), rng.normal(size=n)])
    r[:n] += 5.0 * np.sin(np.arange(n) / 400.0)   # smooth drift inside the first block only
    tau_blocks = residual_autocorr_factor(x, r)
    assert tau_blocks > 1.5


def test_short_residual_vector_gives_one():
    from intensity_fit.fitting import residual_autocorr_factor
    assert residual_autocorr_factor(np.arange(5.0), np.ones(5)) == 1.0


def _spectrum_fit(scn, **opts):
    return run_fit(scn, fit_mode="spectrum", residual_weighting="none", n_mc=400,
                   lineshape={"profile": "gaussian", "fwhm_MHz": 0.08, "window_MHz": 0.3, "normalize": "area"}, **opts)[0]


def _noisy_spectrum_scenario(tmp_path, seed=3):
    scn = build_scenario(tmp_path, [SpeciesSpec("alpha", 3000.0, n_a=5, scale=0.5, T=6.0), SpeciesSpec("beta", 4000.0, n_a=4, scale=2.0, T=6.0)],
                         normalize="area")
    spec = np.loadtxt(scn.spectrum_path)
    rng = np.random.default_rng(seed)
    noise = np.convolve(rng.normal(size=spec.shape[0] + 20), np.ones(13) / 13.0, mode="same")[10:-10]   # correlated over ~13 points
    spec[:, 1] += 0.02 * noise * spec[:, 1].max()
    np.savetxt(scn.spectrum_path, spec, fmt="%.6f %.10e")
    return scn


def test_spectrum_errors_are_scaled_by_the_autocorrelation_factor(tmp_path):
    scn = _noisy_spectrum_scenario(tmp_path)
    on, off = _spectrum_fit(scn), _spectrum_fit(scn, correct_correlated_residuals=False)
    tau = on["fit_stats"]["residual_autocorr_factor"]
    assert tau > 3.0 and on["fit_stats"]["errors_scaled_by_autocorr"] is True
    assert off["fit_stats"]["errors_scaled_by_autocorr"] is False
    for name in ("alpha", "beta"):
        assert on["species"][name]["scale"] == pytest.approx(off["species"][name]["scale"], rel=1e-9)     # central values unchanged
        assert on["species"][name]["scale_err"] / off["species"][name]["scale_err"] == pytest.approx(np.sqrt(tau), rel=1e-6)
        assert on["species"][name]["relative_abundance_mc_rms"] > 2.0 * off["species"][name]["relative_abundance_mc_rms"]
    assert on["temperature_groups"]["ALL"]["Trot_err_K"] / off["temperature_groups"]["ALL"]["Trot_err_K"] == pytest.approx(np.sqrt(tau), rel=1e-6)
    assert np.allclose(np.array(on["covariance_matrix"]), tau * np.array(off["covariance_matrix"]))


def test_line_intensity_errors_are_not_scaled(tmp_path, two_species):
    res, *_ = run_fit(two_species)
    assert np.isnan(res["fit_stats"]["residual_autocorr_factor"]) and res["fit_stats"]["errors_scaled_by_autocorr"] is False


def test_analytic_relative_abundance_error_matches_monte_carlo_for_small_errors(tmp_path):
    scn = _noisy_spectrum_scenario(tmp_path)
    res = _spectrum_fit(scn)
    res_mc = run_fit(scn, fit_mode="spectrum", residual_weighting="none", n_mc=20000, random_seed=5,
                     lineshape={"profile": "gaussian", "fwhm_MHz": 0.08, "window_MHz": 0.3, "normalize": "area"})[0]
    for name in ("alpha", "beta"):
        a = res_mc["species"][name]["relative_abundance_err"]
        m = res_mc["species"][name]["relative_abundance_mc_rms"]
        assert a > 0 and a == pytest.approx(m, rel=0.08)


def test_analytic_relative_abundance_error_is_zero_when_all_scales_are_fixed_but_one(two_species):
    cfg = load_run_config(two_species.config_path)
    opts = FitOptions(**{**cfg.raw["fit"], "fixed_temperature_K": 6.0})
    cfg.species[0].fixed_scale = 0.5
    tables = build_species_tables(cfg.species, opts)
    res = fit_mixture(tables, read_spectrum(cfg.spectrum_path), opts)
    assert res["species"]["alpha"]["scale_fixed"] is True
    assert res["species"]["beta"]["relative_abundance_err"] > 0
    assert res["species"]["alpha"]["relative_abundance_err"] > 0          # depends on the free species through the normalisation


def test_relative_abundance_errors_of_two_species_are_equal(two_species):
    # f1 + f2 = 1, so the two fractions must carry the same uncertainty
    res, *_ = run_fit(two_species, n_mc=0)
    a, b = res["species"]["alpha"]["relative_abundance_err"], res["species"]["beta"]["relative_abundance_err"]
    assert a == pytest.approx(b, rel=1e-6, abs=1e-15)
    assert res["species"]["beta"]["population_relmax_err"] == pytest.approx(0.0, abs=1e-12)   # the reference species has relmax = 1 exactly
