import json

import numpy as np
import pandas as pd
import pytest

from intensity_fit import conformers, workflow
from intensity_fit.config import load_run_config
from intensity_fit.constants import C2
from synth import SpeciesSpec, build_scenario

E0_CM1 = [0.0, 40.0, 90.0, 150.0, 260.0]
TC = 150.0
N_TOTAL = 2.0


def boltzmann_scale(E_cm1, T, g=1.0):
    return N_TOTAL * g * np.exp(-C2 * E_cm1 / T)


def run(tmp_path, model, energies=None, Tc=TC, species_edit=None, extra=(), fit=None):
    """Fit a synthetic mixture whose abundances follow a Boltzmann law at Tc, then analyse it."""
    energies = E0_CM1 if energies is None else energies
    specs = [SpeciesSpec(f"c{i}", 3000.0 + 1000.0 * i, n_a=5, scale=boltzmann_scale(E, Tc), T=6.0)
             for i, E in enumerate(energies)]
    specs += [SpeciesSpec(n, 3000.0 + 1000.0 * (len(specs) + k), n_a=5, scale=s, T=6.0) for k, (n, s) in enumerate(extra)]
    scn = build_scenario(tmp_path, specs, fit={"conformer_population_model": model, **(fit or {})})
    cfg = json.loads(scn.config_path.read_text())
    for item, E in zip(cfg["species"], energies):
        item.update({"dE0": E, "dEe": 1.1 * E})
    if species_edit:
        species_edit(cfg["species"])
    scn.config_path.write_text(json.dumps(cfg))
    return workflow.run_from_config(scn.config_path)


def rows(out, ref):
    return [r for r in out["result"]["conformer_analysis"]["rows"] if r["reference"] == ref]


# ---------- units and settings ----------

@pytest.mark.parametrize("unit, factor", [
    ("cm-1", 1.0), ("CM^-1", 1.0), ("kJ/mol", 83.5934722), ("kcal/mol", 349.7551), ("hartree", 219474.6313632),
    ("Eh", 219474.6313632), ("eV", 8065.543937), ("K", 1 / C2),
])
def test_energy_unit_to_cm1(unit, factor):
    assert conformers.energy_unit_to_cm1(unit) == pytest.approx(factor, rel=1e-6)


def test_unknown_unit_raises():
    with pytest.raises(ValueError, match="Unknown energy unit"):
        conformers.energy_unit_to_cm1("furlong")


def test_tconf_setting_parsing():
    f = conformers._tconf_setting
    assert f({"Tconf": 120}) == ("fixed", 120.0)
    assert f({"Tconf": "fit"}) == ("fit", None)
    assert f({"Tconf": "Mean_Rotational"}) == ("mean_rotational", None)
    assert f({}) == ("mean_rotational", None)
    assert f({"temperature_source": "fixed", "temperature_K": 90}) == ("fixed", 90.0)  # legacy keys
    with pytest.raises(ValueError):
        f({"Tconf": "bogus"})
    with pytest.raises(ValueError):
        f({"Tconf": -5})


def test_reference_parsing():
    assert conformers._references({}) == ["Ee", "E0"]
    assert conformers._references({"reference": "e0"}) == ["E0"]
    assert conformers._references({"reference": "EE"}) == ["Ee"]
    with pytest.raises(ValueError):
        conformers._references({"reference": "x"})


def test_species_config_reads_conformer_keys(tmp_path):
    scn = build_scenario(tmp_path, [SpeciesSpec("a", 3000.0)])
    cfg = json.loads(scn.config_path.read_text())
    cfg["species"][0].update({"dEe": 1.5, "dE0": 1.2, "exclude_from_Tconf_fit": True})
    scn.config_path.write_text(json.dumps(cfg))
    sp = load_run_config(scn.config_path).species[0]
    assert (sp.dEe, sp.dE0, sp.exclude_from_Tconf_fit) == (1.5, 1.2, True)


def test_disabled_analysis_is_empty(tmp_path):
    out = run(tmp_path, {"enabled": False})
    assert out["result"]["conformer_analysis"] == {"rows": [], "fits": {}, "warnings": [], "excluded": [], "groups": {}}
    assert not (tmp_path / "out" / "conformer_populations.csv").exists()


# ---------- fitting Tconf ----------

def test_fit_recovers_boltzmann_temperature(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0"})
    fit = out["result"]["conformer_analysis"]["fits"]["E0"]
    assert fit["Tconf_K"] == pytest.approx(TC, rel=1e-3)
    assert fit["source"] == "fit" and fit["n_species"] == 5 and fit["dof"] == 3
    for r in rows(out, "E0"):
        assert r["fitted_fraction"] == pytest.approx(r["expected_fraction"], rel=1e-3)
        assert r["fitted_over_expected"] == pytest.approx(1.0, rel=1e-3)
    assert sum(r["expected_fraction"] for r in rows(out, "E0")) == pytest.approx(1.0)


def test_fit_is_independent_of_energy_zero(tmp_path):
    def shift(items):
        for it in items:
            it["dE0"] += 500.0

    for name in ("plain", "shifted"):
        (tmp_path / name).mkdir()
    model = {"enabled": True, "Tconf": "fit", "reference": "E0"}
    base = run(tmp_path / "plain", model)
    moved = run(tmp_path / "shifted", model, species_edit=shift)
    assert moved["result"]["conformer_analysis"]["fits"]["E0"]["Tconf_K"] == pytest.approx(
        base["result"]["conformer_analysis"]["fits"]["E0"]["Tconf_K"], rel=1e-6)


def test_fit_both_references_scale_with_the_energy_ratio(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "both"})
    fits = out["result"]["conformer_analysis"]["fits"]
    assert fits["E0"]["Tconf_K"] == pytest.approx(TC, rel=1e-3)
    assert fits["Ee"]["Tconf_K"] == pytest.approx(1.1 * TC, rel=1e-3)  # dEe = 1.1 dE0 -> same ratio E/T
    assert {r["reference"] for r in out["result"]["conformer_analysis"]["rows"]} == {"Ee", "E0"}


def test_energies_in_other_units(tmp_path):
    def to_kj(items):
        for it in items:
            it["dE0"] = it["dE0"] / 83.5934722
            it["dEe"] = it["dEe"] / 83.5934722
    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0", "unit": "kJ/mol"}, species_edit=to_kj)
    assert out["result"]["conformer_analysis"]["fits"]["E0"]["Tconf_K"] == pytest.approx(TC, rel=1e-3)
    assert out["result"]["conformer_analysis"]["unit"] == "kJ/mol"


def test_degeneracy_enters_the_model(tmp_path):
    energies = E0_CM1
    specs = [SpeciesSpec(f"c{i}", 3000.0 + 1000.0 * i, n_a=5, scale=boltzmann_scale(E, TC, g=3.0 if i == 2 else 1.0), T=6.0)
             for i, E in enumerate(energies)]
    scn = build_scenario(tmp_path, specs, fit={"conformer_population_model": {"enabled": True, "Tconf": "fit", "reference": "E0"}})
    cfg = json.loads(scn.config_path.read_text())
    for i, (item, E) in enumerate(zip(cfg["species"], energies)):
        item["dE0"] = E
        if i == 2:
            item["conformer_degeneracy"] = 3
    scn.config_path.write_text(json.dumps(cfg))
    out = workflow.run_from_config(scn.config_path)
    assert out["result"]["conformer_analysis"]["fits"]["E0"]["Tconf_K"] == pytest.approx(TC, rel=1e-3)
    wrong = out["result"]["conformer_analysis"]["rows"][2]["degeneracy"]
    assert wrong == 3.0


def test_fit_reports_uncertainty_and_fraction_errors(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0"})
    fit = out["result"]["conformer_analysis"]["fits"]["E0"]
    assert fit["Tconf_err_K"] >= 0.0
    assert all(r["fitted_fraction_err"] >= 0.0 for r in rows(out, "E0"))


# ---------- fixed / mean temperature ----------

def test_fixed_tconf_uses_the_given_value(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": 150.0, "reference": "E0"})
    info = out["result"]["conformer_analysis"]["fits"]["E0"]
    assert (info["source"], info["Tconf_K"], info["n_species"]) == ("fixed", 150.0, 5)
    assert np.isnan(info["Tconf_err_K"])
    assert {r["Tconf_K"] for r in rows(out, "E0")} == {150.0}
    for r in rows(out, "E0"):
        assert r["fitted_over_expected"] == pytest.approx(1.0, rel=1e-3)


def test_wrong_fixed_tconf_shows_up_in_the_ratio(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": 300.0, "reference": "E0"})
    ratios = [r["fitted_over_expected"] for r in rows(out, "E0")]
    assert max(ratios) / min(ratios) > 1.2


def test_expected_fractions_follow_boltzmann_for_fixed_tconf(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": 80.0, "reference": "E0"})
    w = np.exp(-C2 * np.array(E0_CM1) / 80.0)
    assert [r["expected_fraction"] for r in rows(out, "E0")] == pytest.approx((w / w.sum()).tolist())


def test_mean_rotational_temperature_is_the_default(tmp_path):
    out = run(tmp_path, {"enabled": True, "reference": "E0"})
    assert out["result"]["conformer_analysis"]["fits"]["E0"]["Tconf_K"] == pytest.approx(6.0, rel=1e-3)
    assert out["result"]["conformer_analysis"]["fits"]["E0"]["source"] == "mean_rotational"


# ---------- which species take part ----------

def test_excluded_species_do_not_change_the_fit(tmp_path):
    def flag(items):
        items[-1].update({"dE0": 10.0, "exclude_from_Tconf_fit": True})  # would distort the fit if used

    for name in ("plain", "extra"):
        (tmp_path / name).mkdir()
    model = {"enabled": True, "Tconf": "fit", "reference": "E0"}
    a = run(tmp_path / "plain", model)["result"]["conformer_analysis"]
    b = run(tmp_path / "extra", model, extra=[("complex", 77.0)], species_edit=flag)["result"]["conformer_analysis"]
    assert b["fits"]["E0"]["Tconf_K"] == pytest.approx(a["fits"]["E0"]["Tconf_K"], rel=1e-6)
    assert b["fits"]["E0"]["n_species"] == 5
    assert "complex" not in {r["species"] for r in b["rows"]}
    assert b["excluded"] == ["complex"]


def test_unflagged_extra_species_changes_the_result(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0"}, extra=[("complex", 77.0)],
              species_edit=lambda items: items[-1].update({"dE0": 10.0}))
    assert out["result"]["conformer_analysis"]["fits"]["E0"]["n_species"] == 6
    assert out["result"]["conformer_analysis"]["fits"]["E0"]["Tconf_K"] != pytest.approx(TC, rel=1e-2)


def test_species_without_the_requested_energy_are_skipped(tmp_path):
    def drop(items):
        del items[0]["dEe"]

    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "both"}, species_edit=drop)
    assert len(rows(out, "Ee")) == 4
    assert len(rows(out, "E0")) == 5


def test_species_list_restricts_the_comparison(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0", "species": ["c0", "c1", "c2"]})
    assert {r["species"] for r in rows(out, "E0")} == {"c0", "c1", "c2"}
    assert out["result"]["conformer_analysis"]["fits"]["E0"]["Tconf_K"] == pytest.approx(TC, rel=1e-3)


def test_legacy_delta_e_cm1_feeds_e0(tmp_path):
    def legacy(items):
        for it in items:
            it["delta_E_cm-1"] = it.pop("dE0")
            del it["dEe"]

    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "both"}, species_edit=legacy)
    assert out["result"]["conformer_analysis"]["fits"]["E0"]["Tconf_K"] == pytest.approx(TC, rel=1e-3)
    assert "Ee" not in out["result"]["conformer_analysis"]["fits"]
    assert any("Ee" in w for w in out["result"]["conformer_analysis"]["warnings"])


def test_fixed_scale_species_are_skipped_with_a_warning(tmp_path):
    def fix(items):
        items[1]["fixed_scale"] = boltzmann_scale(E0_CM1[1], TC)

    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0"}, species_edit=fix)
    analysis = out["result"]["conformer_analysis"]
    assert "c1" not in {r["species"] for r in analysis["rows"]}
    assert any("fixed_scale" in w for w in analysis["warnings"])


# ---------- degenerate / failing cases ----------

def test_fit_with_one_species_warns_and_does_not_fit(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0", "species": ["c0"]})
    analysis = out["result"]["conformer_analysis"]
    assert analysis["fits"] == {} and analysis["rows"] == []
    assert any("at least 2 species" in w for w in analysis["warnings"])


def test_fit_on_two_species_has_no_uncertainty(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0", "species": ["c0", "c3"]})
    fit = out["result"]["conformer_analysis"]["fits"]["E0"]
    assert fit["Tconf_K"] == pytest.approx(TC, rel=1e-3)
    assert np.isnan(fit["Tconf_err_K"])
    assert any("degrees of freedom" in w for w in out["result"]["conformer_analysis"]["warnings"])


def test_temperature_on_a_bound_is_flagged(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0", "Tconf_max": 100.0})
    fit = out["result"]["conformer_analysis"]["fits"]["E0"]
    assert fit["Tconf_K"] == pytest.approx(100.0, rel=1e-4)
    assert fit["at_bound"] is True
    assert any("bound" in w for w in out["result"]["warnings"])


def test_no_species_with_energies_warns(tmp_path):
    def wipe(items):
        for it in items:
            it.pop("dE0"), it.pop("dEe")

    out = run(tmp_path, {"enabled": True, "Tconf": "fit"}, species_edit=wipe)
    assert out["result"]["conformer_analysis"]["rows"] == []
    assert len(out["result"]["conformer_analysis"]["warnings"]) == 2


def test_missing_covariance_falls_back_to_log_scale_errors(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0"})
    result = dict(out["result"])
    result.pop("covariance_matrix")
    redone = conformers.analyze_conformers(out["config"].species, out["config"].fit_options, result)
    assert redone["fits"]["E0"]["Tconf_K"] == pytest.approx(TC, rel=1e-3)


def test_missing_uncertainties_gives_unweighted_fit(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0"})
    result = json.loads(json.dumps({k: v for k, v in out["result"].items() if k == "species"}, default=float))
    for v in result["species"].values():
        v["log_scale_err"] = float("nan")
    redone = conformers.analyze_conformers(out["config"].species, out["config"].fit_options, result)
    assert redone["fits"]["E0"]["Tconf_K"] == pytest.approx(TC, rel=1e-3)
    assert any("unweighted" in w for w in redone["warnings"])
    assert np.isnan(redone["rows"][0]["fitted_fraction_err"])


# ---------- outputs ----------

def test_outputs_csv_and_report(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "both"})
    csv = pd.read_csv(tmp_path / "out" / "conformer_populations.csv")
    assert set(csv["reference"]) == {"Ee", "E0"} and len(csv) == 10
    text = (tmp_path / "out" / "synthetic.ifit").read_text()
    assert "CONFORMER POPULATIONS" in text
    assert "E0: Tconf = " in text and "Ee: Tconf = " in text


def test_species_summary_has_both_energies(tmp_path):
    run(tmp_path, {"enabled": False})
    summary = pd.read_csv(tmp_path / "out" / "species_summary.csv")
    assert summary["dE0"].tolist() == pytest.approx(E0_CM1)
    assert summary["dEe"].tolist() == pytest.approx([1.1 * E for E in E0_CM1])


# ---------- numerical checks of the building blocks ----------

def test_degeneracy_changes_the_expected_fractions(tmp_path):
    specs = [SpeciesSpec(f"c{i}", 3000.0 + 1000.0 * i, n_a=5, scale=boltzmann_scale(E, TC, g=3.0 if i == 2 else 1.0), T=6.0)
             for i, E in enumerate(E0_CM1)]
    scn = build_scenario(tmp_path, specs, fit={"conformer_population_model": {"enabled": True, "Tconf": TC, "reference": "E0"}})
    cfg = json.loads(scn.config_path.read_text())
    for i, (item, E) in enumerate(zip(cfg["species"], E0_CM1)):
        item["dE0"] = E
        if i == 2:
            item["conformer_degeneracy"] = 3
    scn.config_path.write_text(json.dumps(cfg))
    out = workflow.run_from_config(scn.config_path)
    w = np.array([3.0 if i == 2 else 1.0 for i in range(5)]) * np.exp(-C2 * np.array(E0_CM1) / TC)
    assert [r["expected_fraction"] for r in rows(out, "E0")] == pytest.approx((w / w.sum()).tolist())
    assert [r["fitted_over_expected"] for r in rows(out, "E0")] == pytest.approx([1.0] * 5, rel=1e-3)


def test_fit_tconf_matches_ordinary_least_squares_with_its_error():
    rng = np.random.default_rng(3)
    E = np.array([0.0, 30.0, 70.0, 120.0, 200.0, 310.0])
    x = C2 * E
    beta_true, c_true = 1.0 / 120.0, 0.7
    y = c_true - x * beta_true + rng.normal(0.0, 0.05, E.size)
    fit = conformers._fit_tconf(x, y, np.eye(E.size), T_start=300.0, T_min=1.0, T_max=1e5)

    slope, intercept = np.polyfit(x, y, 1)
    beta = -slope
    s = np.sqrt(np.sum((y - (intercept + slope * x)) ** 2) / (E.size - 2))
    beta_err = s / np.sqrt(np.sum((x - x.mean()) ** 2))
    assert fit["Tconf_K"] == pytest.approx(1.0 / beta, rel=1e-6)
    assert fit["c"] == pytest.approx(intercept, rel=1e-6)
    assert fit["Tconf_err_K"] == pytest.approx(beta_err / beta ** 2, rel=1e-4)
    assert fit["chi2_red"] == pytest.approx(s ** 2, rel=1e-6)
    assert fit["dof"] == 4 and fit["at_bound"] is False


def test_fit_tconf_whitening_downweights_noisy_points():
    E = np.array([0.0, 50.0, 100.0, 200.0])
    x = C2 * E
    y = 0.3 - x / 150.0
    y_bad = y.copy()
    y_bad[3] += 5.0
    cov = np.diag([1e-4, 1e-4, 1e-4, 1e4])  # last point is essentially unconstrained
    whiten = conformers._whitener(cov, 4)
    weighted = conformers._fit_tconf(x, y_bad, whiten, 300.0, 1.0, 1e5)
    plain = conformers._fit_tconf(x, y_bad, np.eye(4), 300.0, 1.0, 1e5)
    assert weighted["Tconf_K"] == pytest.approx(150.0, rel=1e-3)
    assert abs(plain["Tconf_K"] - 150.0) > 5.0


def test_whitener_falls_back_for_non_positive_definite_covariance():
    cov = np.array([[1.0, 2.0], [2.0, 1.0]])  # not PD, diagonal still positive
    assert conformers._whitener(cov, 2) == pytest.approx(np.eye(2))
    assert conformers._whitener(None, 3) == pytest.approx(np.eye(3))
    assert conformers._whitener(np.array([[-1.0, 0.0], [0.0, 1.0]]), 2) == pytest.approx(np.eye(2))


def test_fraction_errors_match_monte_carlo_propagation():
    from intensity_fit.config import FitOptions, SpeciesConfig

    names = ["a", "b", "c"]
    scales = np.array([1.0, 0.6, 0.25])
    cov = np.array([[0.010, 0.004, 0.0], [0.004, 0.020, 0.002], [0.0, 0.002, 0.030]])  # correlated ln A
    result = {
        "species": {n: {"scale": s, "scale_fixed": False, "log_scale_err": np.sqrt(cov[i, i]), "Trot_K": 5.0}
                    for i, (n, s) in enumerate(zip(names, scales))},
        "covariance_matrix": np.pad(cov, ((0, 1), (0, 1))).tolist(),  # extra row: a temperature parameter
    }
    cfgs = [SpeciesConfig(n, "l", "c", dE0=E) for n, E in zip(names, [0.0, 50.0, 120.0])]
    opts = FitOptions(conformer_population_model={"enabled": True, "Tconf": 100.0, "reference": "E0"})
    rows_ = conformers.analyze_conformers(cfgs, opts, result)["rows"]

    samples = np.exp(np.random.default_rng(1).multivariate_normal(np.log(scales), cov, size=400_000))
    frac = samples / samples.sum(axis=1, keepdims=True)
    assert [r["fitted_fraction"] for r in rows_] == pytest.approx((scales / scales.sum()).tolist())
    assert [r["fitted_fraction_err"] for r in rows_] == pytest.approx(frac.std(axis=0).tolist(), rel=0.05)


# ---------- species groups ----------

def _label_groups(items):
    for i, it in enumerate(items):
        it["group"] = "monomers" if i < 3 else "complexes"


def test_group_key_is_read_and_written_to_the_summary(tmp_path):
    run(tmp_path, {"enabled": False}, species_edit=_label_groups)
    summary = pd.read_csv(tmp_path / "out" / "species_summary.csv")
    assert summary["group"].tolist() == ["monomers"] * 3 + ["complexes"] * 2


def test_groups_option_selects_species(tmp_path):
    model = {"enabled": True, "Tconf": "fit", "reference": "E0"}
    out = run(tmp_path, {**model, "groups": ["monomers"]}, species_edit=_label_groups)
    assert {r["species"] for r in rows(out, "E0")} == {"c0", "c1", "c2"}
    assert out["result"]["conformer_analysis"]["fits"]["E0"]["n_species"] == 3


def test_groups_option_accepts_a_single_string_and_combines_with_exclusion(tmp_path):
    def edit(items):
        _label_groups(items)
        items[0]["exclude_from_Tconf_fit"] = True

    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0", "groups": "monomers"}, species_edit=edit)
    assert {r["species"] for r in rows(out, "E0")} == {"c1", "c2"}


def test_species_without_group_are_ignored_when_groups_is_set(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0", "groups": ["monomers"]})
    assert out["result"]["conformer_analysis"]["rows"] == []


# ---------- labels (names used in the article) ----------

def _label(items):
    for i, it in enumerate(items):
        it["label"] = f"article-{i}"


def test_label_defaults_to_the_species_name(tmp_path):
    out = run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0"})
    assert [r["label"] for r in rows(out, "E0")] == [f"c{i}" for i in range(5)]
    summary = pd.read_csv(tmp_path / "out" / "species_summary.csv")
    assert summary["label"].tolist() == summary["species"].tolist()


def test_label_is_written_to_csv_report_and_summary(tmp_path):
    run(tmp_path, {"enabled": True, "Tconf": "fit", "reference": "E0"}, species_edit=_label)
    summary = pd.read_csv(tmp_path / "out" / "species_summary.csv")
    assert summary["label"].tolist() == [f"article-{i}" for i in range(5)]
    csv = pd.read_csv(tmp_path / "out" / "conformer_populations.csv")
    assert csv["label"].tolist() == [f"article-{i}" for i in range(5)]
    text = (tmp_path / "out" / "synthetic.ifit").read_text()
    assert text.count("article-0") >= 2  # species table and conformer table
    assert "label" in text


def test_label_does_not_change_the_fit(tmp_path):
    for name in ("plain", "labelled"):
        (tmp_path / name).mkdir()
    model = {"enabled": True, "Tconf": "fit", "reference": "E0"}
    a = run(tmp_path / "plain", model)["result"]["conformer_analysis"]["fits"]["E0"]["Tconf_K"]
    b = run(tmp_path / "labelled", model, species_edit=_label)["result"]["conformer_analysis"]["fits"]["E0"]["Tconf_K"]
    assert a == b


# ---------- abundance relative to the species group, and one Tconf analysis per group ----------

TC2 = 80.0
DE_PAIR = 60.0   # cm-1 between the two "complexes"


def run_groups(tmp_path, model, mono_group="mono", cplx_group="cplx", edit=None, fit=None):
    """Five monomer-like species (Boltzmann at 150 K) and a pair of complex-like species (Boltzmann at 80 K)."""
    complexes = [("x0", N_TOTAL), ("x1", N_TOTAL * np.exp(-C2 * DE_PAIR / TC2))]

    def label(items):
        for i, it in enumerate(items):
            if i < 5:
                it["group"] = mono_group
            else:
                it["group"] = cplx_group
                it.update({"dE0": 0.0 if i == 5 else DE_PAIR, "dEe": 0.0 if i == 5 else DE_PAIR})
        if edit:
            edit(items)

    return run(tmp_path, {"enabled": True, "reference": "E0", **model}, extra=complexes, species_edit=label, fit=fit)


def test_group_fractions_sum_to_one_inside_each_group_and_skip_ungrouped(tmp_path):
    def ungroup(items):
        items[0].pop("group", None)

    run_groups(tmp_path, {"per_group": False}, edit=ungroup)
    summary = pd.read_csv(tmp_path / "out" / "species_summary.csv")
    assert summary[summary.group == "mono"]["group_fraction"].sum() == pytest.approx(1.0)
    assert summary[summary.group == "cplx"]["group_fraction"].sum() == pytest.approx(1.0)
    assert (summary["group_fraction_err"].dropna() >= 0).all()
    assert pd.isna(summary.loc[0, "group_fraction"]) and pd.isna(summary.loc[0, "group"])


def test_group_fraction_matches_the_scales_and_the_delta_method(tmp_path):
    out = run_groups(tmp_path, {"per_group": False})
    res, cfg = out["result"], out["config"]
    gf = conformers.group_fractions(cfg.species, res)
    names = [sp.name for sp in cfg.species if sp.group == "mono"]
    A = np.array([res["species"][n]["scale"] for n in names])
    assert [gf[n][0] for n in names] == pytest.approx((A / A.sum()).tolist())
    # independent check of the error with a sample of the covariance of ln(scale)
    all_names = list(res["species"])
    C = np.array(res["covariance_matrix"])[: len(all_names), : len(all_names)]
    idx = [all_names.index(n) for n in names]
    S = np.exp(np.random.default_rng(0).multivariate_normal(np.log(A), C[np.ix_(idx, idx)], size=300000))
    F = S / S.sum(axis=1, keepdims=True)
    assert [gf[n][1] for n in names] == pytest.approx(F.std(axis=0).tolist(), rel=0.1, abs=1e-12)


def test_group_fraction_is_na_in_the_report_without_a_group(tmp_path):
    run(tmp_path, {"enabled": False})
    text = (tmp_path / "out" / "synthetic.ifit").read_text()
    row = [l for l in text.split("FINAL SPECIES RESULTS")[1].splitlines() if l.startswith(" c0 ")][0]
    assert row.count(" NA") >= 2
    assert "grp_%" in text and "+/-relmax" in text


def test_per_group_fits_a_separate_tconf_for_every_group(tmp_path):
    out = run_groups(tmp_path, {"per_group": True, "Tconf": "fit"})
    ana = out["result"]["conformer_analysis"]
    assert set(ana["groups"]) == {"mono", "cplx"}
    assert ana["groups"]["mono"]["fits"]["E0"]["Tconf_K"] == pytest.approx(TC, rel=1e-3)
    assert ana["groups"]["cplx"]["fits"]["E0"]["Tconf_K"] == pytest.approx(TC2, rel=1e-3)
    assert {r["group"] for r in ana["rows"]} == {"mono", "cplx"}
    for g in ("mono", "cplx"):                       # every group is normalised by itself
        assert sum(r["fitted_fraction"] for r in ana["rows"] if r["group"] == g) == pytest.approx(1.0)
        assert sum(r["expected_fraction"] for r in ana["rows"] if r["group"] == g) == pytest.approx(1.0)
    assert any("degrees of freedom" in w and "[cplx]" in w for w in ana["warnings"])    # two species: exactly determined


def test_per_group_false_keeps_a_single_analysis_over_every_species(tmp_path):
    out = run_groups(tmp_path, {"per_group": False, "Tconf": "fit"})
    ana = out["result"]["conformer_analysis"]
    assert ana["groups"] == {}
    assert ana["fits"]["E0"]["n_species"] == 7
    assert all(r["group"] == "" for r in ana["rows"])


def test_tconf_from_another_group_is_used_without_fitting(tmp_path):
    out = run_groups(tmp_path, {"per_group": True, "Tconf": "fit", "group_options": {"cplx": {"Tconf": "from:mono"}}})
    ana = out["result"]["conformer_analysis"]
    mono_T = ana["groups"]["mono"]["fits"]["E0"]["Tconf_K"]
    cplx = ana["groups"]["cplx"]["fits"]["E0"]
    assert cplx["Tconf_K"] == pytest.approx(mono_T) and cplx["source"] == "from mono"
    assert np.isnan(cplx["Tconf_err_K"])
    rows_c = [r for r in ana["rows"] if r["group"] == "cplx"]
    assert {r["Tconf_source"] for r in rows_c} == {"from mono"}
    w = np.array([1.0, np.exp(-C2 * DE_PAIR / mono_T)])
    assert [r["expected_fraction"] for r in rows_c] == pytest.approx((w / w.sum()).tolist())
    ratio = rows_c[1]["fitted_fraction"] / rows_c[0]["fitted_fraction"]                 # fitted fractions are unchanged
    assert ratio == pytest.approx(np.exp(-C2 * DE_PAIR / TC2), rel=1e-3)
    assert not np.isclose(rows_c[1]["fitted_over_expected"], 1.0, atol=0.02)         # the two temperatures disagree


def test_from_group_pointing_to_a_missing_group_warns_and_skips(tmp_path):
    out = run_groups(tmp_path, {"per_group": True, "Tconf": "fit", "group_options": {"cplx": {"Tconf": "from:nowhere"}}})
    ana = out["result"]["conformer_analysis"]
    assert "cplx" not in ana["groups"]
    assert any("nowhere" in w for w in ana["warnings"])


def test_groups_option_chooses_which_groups_are_analysed(tmp_path):
    out = run_groups(tmp_path, {"per_group": True, "Tconf": "fit", "groups": ["mono"]})
    assert set(out["result"]["conformer_analysis"]["groups"]) == {"mono"}


def test_group_options_override_tconf_and_reference(tmp_path):
    out = run_groups(tmp_path, {"per_group": True, "Tconf": "fit", "group_options": {"cplx": {"Tconf": 100.0, "reference": "Ee"}}})
    ana = out["result"]["conformer_analysis"]
    assert ana["groups"]["cplx"]["fits"]["Ee"]["Tconf_K"] == 100.0 and "E0" not in ana["groups"]["cplx"]["fits"]
    assert ana["groups"]["mono"]["fits"]["E0"]["Tconf_K"] == pytest.approx(TC, rel=1e-3)


def test_group_without_energies_is_skipped_with_a_warning(tmp_path):
    def strip(items):
        for it in items[5:]:
            it.pop("dE0", None); it.pop("dEe", None)

    out = run_groups(tmp_path, {"per_group": True, "Tconf": "fit"}, edit=strip)
    ana = out["result"]["conformer_analysis"]
    assert set(ana["groups"]) == {"mono"}
    assert any("[cplx]" in w and "no energies" in w for w in ana["warnings"])


def test_exclusion_flag_still_applies_inside_a_group(tmp_path):
    def flag(items):
        items[4]["exclude_from_Tconf_fit"] = True

    out = run_groups(tmp_path, {"per_group": True, "Tconf": "fit"}, edit=flag)
    ana = out["result"]["conformer_analysis"]
    assert ana["groups"]["mono"]["fits"]["E0"]["n_species"] == 4
    assert "c4" not in {r["species"] for r in ana["rows"]}


def test_report_lists_every_group_and_its_tconf(tmp_path):
    run_groups(tmp_path, {"per_group": True, "Tconf": "fit", "group_options": {"cplx": {"Tconf": "from:mono"}}})
    text = (tmp_path / "out" / "synthetic.ifit").read_text()
    assert " mono: E0: Tconf = " in text and " cplx: E0: Tconf = " in text and "(from mono, 2 species)" in text
    assert " GROUP: mono   ref: E0   Tconf = " in text and "RMS for mono E0 at Tconf" in text
    csv = pd.read_csv(tmp_path / "out" / "conformer_populations.csv")
    assert set(csv["group"]) == {"mono", "cplx"} and len(csv) == 7


def test_also_with_tconf_from_adds_rows_with_the_other_groups_temperature(tmp_path):
    out = run_groups(tmp_path, {"per_group": True, "Tconf": "fit", "group_options": {"cplx": {"also_with_Tconf_from": "mono"}}})
    ana = out["result"]["conformer_analysis"]
    mono_T = ana["groups"]["mono"]["fits"]["E0"]["Tconf_K"]
    assert ana["groups"]["cplx"]["fits"]["E0"]["Tconf_K"] == pytest.approx(TC2, rel=1e-3)               # own fit is kept
    extra = ana["groups"]["cplx (Tconf from mono)"]["fits"]["E0"]
    assert extra["Tconf_K"] == pytest.approx(mono_T) and extra["source"] == "from mono"
    rows_c = [r for r in ana["rows"] if r["group"] == "cplx"]
    assert len(rows_c) == 4                                                                        # 2 species x (own Tconf, other Tconf)
    own = [r for r in rows_c if r["Tconf_source"] == "fit"]
    other = [r for r in rows_c if r["Tconf_source"] == "from mono"]
    assert [r["fitted_fraction"] for r in own] == pytest.approx([r["fitted_fraction"] for r in other])
    assert [r["expected_fraction"] for r in own] != pytest.approx([r["expected_fraction"] for r in other])
    assert all(r["fitted_over_expected"] == pytest.approx(1.0, rel=1e-3) for r in own)


def test_also_with_tconf_from_unknown_group_warns(tmp_path):
    out = run_groups(tmp_path, {"per_group": True, "Tconf": "fit", "group_options": {"cplx": {"also_with_Tconf_from": "nowhere"}}})
    ana = out["result"]["conformer_analysis"]
    assert len([r for r in ana["rows"] if r["group"] == "cplx"]) == 2
    assert any("nowhere" in w for w in ana["warnings"])


def test_report_has_a_tconf_column(tmp_path):
    run_groups(tmp_path, {"per_group": True, "Tconf": "fit", "group_options": {"cplx": {"also_with_Tconf_from": "mono"}}})
    text = (tmp_path / "out" / "synthetic.ifit").read_text()
    assert "GROUP: cplx   ref: E0   Tconf = " in text and "(from mono)" in text and "cplx (Tconf from mono): E0: Tconf = " in text


def test_report_prints_energies_in_the_selected_unit(tmp_path):
    run_groups(tmp_path, {"per_group": True, "Tconf": "fit", "unit": "kJ/mol"})
    text = (tmp_path / "out" / "synthetic.ifit").read_text()
    assert "E/kJ/mol" in text and "E/cm-1" not in text
    conf_part = text.split("CONFORMER POPULATIONS")[1]
    shown = [float(l.split()[2]) for l in conf_part.splitlines() if l.startswith(" x1 ")]
    assert shown and shown[0] == pytest.approx(DE_PAIR, abs=1e-6)   # input dE are read in the unit, so they are printed back unchanged
