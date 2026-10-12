import json

import numpy as np
import pandas as pd
import pytest

from intensity_fit import cli, workflow
from intensity_fit.config import load_run_config
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


def test_int_dipole_warning_reaches_the_report_and_the_result(tmp_path):
    scn = build_scenario(tmp_path, [SpeciesSpec("dup", 3000.0, n_a=3)])
    (tmp_path / "dup.int").write_text(
        "dup\n0 1 1.0 0 40 -10.0 -1.0 100.0 20.0 25\n 001 1.0 / a dipole\n 001 0.5 / a dipole\n 003 0.2 / c dipole\n")
    out = workflow.run_from_config(scn.config_path)
    assert any("dup: .int dipole ID 001 is listed 2 times" in w for w in out["result"]["warnings"])
    text = (scn.out_dir / "synthetic.ifit").read_text()
    assert "WARNING: dipole ID 001 is listed 2 times" in text
    assert out["species_tables"]["dup"].attrs["dipoles"]["a"] == pytest.approx(1.5)


def _worst_lines(report_text):
    section = report_text.split("WORST LINES BY ABSOLUTE RESIDUAL")[1].split("WARNINGS")[0].strip().splitlines()[1:]
    rows = []
    for line in section:
        parts = line.split()
        rows.append((float(parts[2]), float(parts[3]), float(parts[4])))  # exp, calc, residual
    return rows


@pytest.mark.parametrize("fit", [
    {"fit_mode": "line_intensity"},
    {"fit_mode": "spectrum", "residual_weighting": "none",
     "lineshape": {"profile": "gaussian", "fwhm_MHz": 0.08, "window_MHz": 0.3, "normalize": "area"}},
])
def test_worst_lines_table_is_self_consistent_and_sums_blends(tmp_path, fit):
    # two species share J = 0..3 frequencies; the spectrum is built perturbed so that residuals are not zero
    specs = [SpeciesSpec("a", 3000.0, n_a=5, scale=0.5), SpeciesSpec("b", 3000.0, n_a=4, scale=2.0), SpeciesSpec("c", 4000.0, n_a=4, scale=1.0)]
    scn = build_scenario(tmp_path, specs, fit=fit, normalize="area" if fit["fit_mode"] == "spectrum" else "peak")
    spec = np.loadtxt(scn.spectrum_path)
    spec[(spec[:, 0] > 3190.0) & (spec[:, 0] < 3210.0), 1] *= 1.7      # a disturbed blended peak (J = 2 of a and b)
    spec[(spec[:, 0] > 4290.0) & (spec[:, 0] < 4310.0), 1] *= 0.5      # a disturbed isolated peak
    np.savetxt(scn.spectrum_path, spec, fmt="%.6f %.10e")
    workflow.run_from_config(scn.config_path)
    rows = _worst_lines((scn.out_dir / "synthetic.ifit").read_text())
    assert rows
    for exp, calc, residual in rows:
        assert abs(exp - calc - residual) <= 1e-5 * max(abs(exp), abs(calc)) + 1e-12  # the report prints 7 digits
    text = (scn.out_dir / "synthetic.ifit").read_text()
    assert "[+1 blended]" in text


def test_report_and_summary_show_errors_on_scales_and_abundances(tmp_path):
    scn = build_scenario(tmp_path, [SpeciesSpec("alpha", 3000.0, n_a=5, scale=0.5), SpeciesSpec("beta", 4000.0, n_a=4, scale=2.0)],
                         fit={"n_mc": 300})
    out = workflow.run_from_config(scn.config_path)
    summary = pd.read_csv(scn.out_dir / "species_summary.csv")
    assert {"scale_err", "relative_abundance_err", "population_relmax_err"} <= set(summary.columns)
    assert (summary["scale_err"] >= 0).all() and (summary["relative_abundance_err"] >= 0).all()
    text = (scn.out_dir / "synthetic.ifit").read_text()
    assert "+/-scale" in text and "+/-rel_%" in text


def test_spectrum_report_states_the_autocorrelation_factor(tmp_path):
    scn = build_scenario(tmp_path, [SpeciesSpec("alpha", 3000.0, n_a=5, scale=0.5)], normalize="area",
                         fit={"fit_mode": "spectrum", "residual_weighting": "none",
                              "lineshape": {"profile": "gaussian", "fwhm_MHz": 0.08, "window_MHz": 0.3, "normalize": "area"}})
    out = workflow.run_from_config(scn.config_path)
    assert np.isfinite(out["result"]["fit_stats"]["residual_autocorr_factor"])
    assert "resid.autocorr tau=" in (scn.out_dir / "synthetic.ifit").read_text()
