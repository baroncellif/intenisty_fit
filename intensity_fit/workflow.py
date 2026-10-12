from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunConfig, load_run_config
from .conformers import analyze_conformers, group_fractions
from .fitting import fit_mixture
from .pickett_meta import make_partition_function
from .plotting import show_interactive_overlay_plot, write_quick_overlay_plot, write_species_line_pdf
from .reports import print_console_summary, write_ifit_report
from .spectrum import read_spectrum
from .tables import build_line_diagnostic_table, build_species_diagnostic_summary, build_species_tables


def run_from_config(config_path: str | Path) -> dict:
    cfg = load_run_config(config_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    species_tables = build_species_tables(cfg.species, cfg.fit_options)
    spectrum_df = read_spectrum(cfg.spectrum_path)
    q_functions = _build_q_functions(species_tables)

    result = fit_mixture(
        species_tables=species_tables,
        spectrum_df=spectrum_df,
        fit_options=cfg.fit_options,
        q_functions=q_functions,
    )
    conformers = analyze_conformers(cfg.species, cfg.fit_options, result)
    result["conformer_analysis"] = conformers
    result["conformer_population_diagnostics"] = conformers["rows"]
    int_warnings = [
        f"{name}: .int {w}" for name, tab in species_tables.items() for w in tab.attrs.get("int_metadata", {}).get("warnings", [])
    ]
    result["warnings"] = list(result.get("warnings", [])) + conformers["warnings"] + int_warnings

    line_diag = build_line_diagnostic_table(species_tables, spectrum_df, result, cfg.fit_options, q_functions=q_functions)
    species_summary = build_species_diagnostic_summary(species_tables, result, line_diag_df=line_diag)
    gf = group_fractions(cfg.species, result)   # abundance relative to the sum over the species' own group
    species_summary["group_fraction"] = species_summary["species"].map(lambda n: gf.get(n, (np.nan, np.nan))[0])
    species_summary["group_fraction_err"] = species_summary["species"].map(lambda n: gf.get(n, (np.nan, np.nan))[1])

    report_path = output_dir / f"{cfg.name}.ifit"
    write_ifit_report(report_path, cfg, result, species_summary, line_diag, species_tables=species_tables)

    species_summary.to_csv(output_dir / "species_summary.csv", index=False, float_format="%.10g")
    line_diag.to_csv(output_dir / "line_diagnostics.csv", index=False, float_format="%.10g")
    result["observation_table"].to_csv(output_dir / "observations.csv", index=False, float_format="%.10g")
    if conformers["rows"]:
        pd.DataFrame(conformers["rows"]).to_csv(output_dir / "conformer_populations.csv", index=False, float_format="%.10g")

    plot_cfg = dict(cfg.fit_options.plot or cfg.plot or {})
    quick_plot = None
    pdf_path = None
    show_interactive = False
    if plot_cfg.get("enabled", True):
        fmin = plot_cfg.get("fmin")
        fmax = plot_cfg.get("fmax")
        show_interactive = bool(plot_cfg.get("interactive", True))
        if plot_cfg.get("overview_png", False):
            quick_plot = output_dir / f"{cfg.name}_overview.png"
            write_quick_overlay_plot(quick_plot, spectrum_df, line_diag, cfg.fit_options, result=result, fmin=fmin, fmax=fmax)
        if plot_cfg.get("species_pdf", True):
            pdf_path = output_dir / f"{cfg.name}_line_report.pdf"
            write_species_line_pdf(pdf_path, spectrum_df, line_diag, cfg.fit_options, result=result)

    with (output_dir / "config_used.json").open("w") as f:
        json.dump(cfg.raw, f, indent=2)

    print_console_summary(cfg, result, species_summary, report_path)
    if quick_plot is not None:
        print(f"overview  : {quick_plot}")
    if pdf_path is not None:
        print(f"line PDF  : {pdf_path}")
    if show_interactive:
        print("interactive plot: opening Matplotlib window with final fitted model")
        show_interactive_overlay_plot(spectrum_df, line_diag, cfg.fit_options, result=result, fmin=fmin, fmax=fmax)
    return {
        "config": cfg,
        "species_tables": species_tables,
        "spectrum_df": spectrum_df,
        "result": result,
        "line_diag_df": line_diag,
        "species_summary_df": species_summary,
        "report_path": report_path,
        "overview_plot_path": quick_plot,
        "line_pdf_path": pdf_path,
    }


def _build_q_functions(species_tables: dict) -> dict:
    q_functions = {}
    for name, tab in species_tables.items():
        q_table = tab.attrs.get("partition_table")
        qfunc = make_partition_function(q_table)
        if qfunc is not None:
            q_functions[name] = qfunc
    return q_functions
