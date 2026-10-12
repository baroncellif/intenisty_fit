import json

import numpy as np
import pandas as pd
import pytest

from intensity_fit import cli, workflow
from intensity_fit.config import FitOptions, RunConfig, SpeciesConfig, load_run_config
from intensity_fit.constants import C2
from intensity_fit.tables import build_species_tables
from synth import SpeciesSpec, build_scenario


def _enable_outputs(scn, **plot):
    cfg = json.loads(scn.config_path.read_text())
    cfg["plot"] = {"enabled": True, "interactive": False, **plot}
    scn.config_path.write_text(json.dumps(cfg))


def test_run_from_config_end_to_end(two_species, capsys):
    out = workflow.run_from_config(two_species.config_path)
    d = two_species.out_dir
    for name in ["synthetic.ifit", "species_summary.csv", "line_diagnostics.csv", "observations.csv", "config_used.json"]:
        assert (d / name).exists(), name
    assert out["result"]["success"]
    assert out["report_path"] == d / "synthetic.ifit"
    assert out["overview_plot_path"] is None and out["line_pdf_path"] is None
    assert json.loads((d / "config_used.json").read_text()) == two_species.config

    summary = pd.read_csv(d / "species_summary.csv").set_index("species")
    assert summary.loc["alpha", "scale"] == pytest.approx(0.5, rel=1e-4)
    assert summary.loc["beta", "Trot_K"] == pytest.approx(6.0, rel=1e-4)
    assert summary["n_lines"].tolist() == [5, 4]

    diag = pd.read_csv(d / "line_diagnostics.csv")
    assert len(diag) == 9
    assert diag["abs_residual"].max() < 1e-5
    assert len(pd.read_csv(d / "observations.csv")) == 9

    assert "synthetic" in capsys.readouterr().out


def test_report_contents(two_species):
    workflow.run_from_config(two_species.config_path)
    text = (two_species.out_dir / "synthetic.ifit").read_text()
    assert "INTFIT" in text and "FIT QUALITY" in text and "OPTIONS" in text
    assert "alpha" in text and "beta" in text


def test_run_creates_missing_output_directory(two_species):
    assert not two_species.out_dir.exists()
    workflow.run_from_config(two_species.config_path)
    assert two_species.out_dir.is_dir()


def test_overview_png_and_species_pdf_are_written(two_species):
    _enable_outputs(two_species, overview_png=True, species_pdf=True, fmin=2900.0, fmax=4500.0)
    out = workflow.run_from_config(two_species.config_path)
    assert out["overview_plot_path"].stat().st_size > 0
    assert out["line_pdf_path"].stat().st_size > 0


def test_plot_disabled_writes_no_figures(two_species):
    workflow.run_from_config(two_species.config_path)
    assert not list(two_species.out_dir.glob("*.png")) and not list(two_species.out_dir.glob("*.pdf"))


def test_run_with_second_step_dipole_fit(tmp_path):
    specs = [SpeciesSpec("alpha", 3000.0, n_a=5, scale=0.8), SpeciesSpec("beta", 4000.0, n_a=4, n_c=4, mu_scale_c=1.3)]
    scn = build_scenario(tmp_path, specs, fit={"fixed_temperature_K": 6.0, "dipole_model": "component_scaled",
                                                "dipole_fit_second_step": True})
    cfg = json.loads(scn.config_path.read_text())
    cfg["species"][1].update({"fixed_scale": 1.0, "fit_dipole_components": ["c"]})
    scn.config_path.write_text(json.dumps(cfg))
    out = workflow.run_from_config(scn.config_path)
    assert out["result"]["dipole_scales"]["beta:mu_c"]["scale"] == pytest.approx(1.3, rel=1e-4)
    assert "beta:mu_c" in (scn.out_dir / "synthetic.ifit").read_text() or "mu_c" in (scn.out_dir / "synthetic.ifit").read_text()
    diag = pd.read_csv(scn.out_dir / "line_diagnostics.csv")
    assert diag["abs_residual"].max() < 1e-5  # diagnostics reuse the fitted mu scale


def test_build_q_functions(two_species):
    cfg = load_run_config(two_species.config_path)
    tables = build_species_tables(cfg.species, cfg.fit_options)
    q = workflow._build_q_functions(tables)
    assert set(q) == {"alpha", "beta"}
    assert q["alpha"](10.0) == pytest.approx(10.0 ** 1.5)


def test_build_q_functions_skips_species_without_out_file(tmp_path):
    scn = build_scenario(tmp_path, [SpeciesSpec("noout", 3000.0, n_a=3, with_out=False)])
    cfg = load_run_config(scn.config_path)
    assert workflow._build_q_functions(build_species_tables(cfg.species, cfg.fit_options)) == {}


# ---------- conformer population diagnostics ----------

def _cfg(model, species):
    return RunConfig(name="r", spectrum_path="s", output_dir="o", species=species,
                     fit_options=FitOptions(conformer_population_model=model))


def _result(**abund):
    return {"species": {n: {"relative_abundance": a, "Trot_K": 5.0} for n, a in abund.items()}}


def test_conformer_diagnostics_disabled():
    cfg = _cfg(None, [SpeciesConfig("a", "l", "c")])
    assert workflow._conformer_population_diagnostics(cfg, _result(a=1.0), {}) == []
    cfg = _cfg({"enabled": False}, [SpeciesConfig("a", "l", "c")])
    assert workflow._conformer_population_diagnostics(cfg, _result(a=1.0), {}) == []


def test_conformer_diagnostics_boltzmann_with_fixed_temperature():
    species = [SpeciesConfig("a", "l", "c", delta_E_cm1=0.0), SpeciesConfig("b", "l", "c", delta_E_cm1=10.0, conformer_degeneracy=2.0)]
    cfg = _cfg({"enabled": True, "temperature_source": "fixed", "temperature_K": 20.0}, species)
    rows = workflow._conformer_population_diagnostics(cfg, _result(a=0.7, b=0.3), {})
    wb = 2.0 * np.exp(-C2 * 10.0 / 20.0)
    assert rows[0]["expected_fraction"] == pytest.approx(1.0 / (1.0 + wb))
    assert rows[1]["expected_fraction"] == pytest.approx(wb / (1.0 + wb))
    assert sum(r["expected_fraction"] for r in rows) == pytest.approx(1.0)
    assert [r["fitted_fraction"] for r in rows] == [0.7, 0.3]
    assert rows[0]["temperature_K"] == 20.0


def test_conformer_diagnostics_uses_mean_rotational_temperature_by_default():
    species = [SpeciesConfig("a", "l", "c", delta_E_cm1=0.0), SpeciesConfig("b", "l", "c", delta_E_cm1=5.0)]
    res = {"species": {"a": {"relative_abundance": 0.5, "Trot_K": 4.0}, "b": {"relative_abundance": 0.5, "Trot_K": 8.0}}}
    rows = workflow._conformer_population_diagnostics(_cfg({"enabled": True}, species), res, {})
    assert rows[0]["temperature_K"] == pytest.approx(6.0)


def test_conformer_diagnostics_reference_species_defaults_to_zero_energy():
    species = [SpeciesConfig("ref", "l", "c"), SpeciesConfig("x", "l", "c")]
    rows = workflow._conformer_population_diagnostics(
        _cfg({"enabled": True, "reference_species": "ref"}, species), _result(ref=0.9, x=0.1), {})
    assert rows[0]["expected_fraction"] == pytest.approx(1.0)  # x has unknown energy -> excluded
    assert np.isnan(rows[1]["expected_fraction"])


# ---------- CLI ----------

def test_cli_run_returns_zero(two_species, capsys):
    assert cli.main(["run", str(two_species.config_path)]) == 0
    assert (two_species.out_dir / "synthetic.ifit").exists()


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_cli_rejects_unknown_subcommand():
    with pytest.raises(SystemExit):
        cli.main(["bogus"])


def test_cli_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.main(["run", str(tmp_path / "nope.json")])


def test_package_exports():
    import intensity_fit
    for name in intensity_fit.__all__:
        assert hasattr(intensity_fit, name)
