from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunConfig


def _yn(path: str | None) -> str:
    return "yes" if path and Path(path).exists() else "no"


def _fmt_mu(value) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.6f}"


def write_ifit_report(
    path: str | Path,
    cfg: RunConfig,
    result: dict,
    species_summary: pd.DataFrame,
    line_diag: pd.DataFrame,
    species_tables: dict | None = None,
) -> None:
    path = Path(path)
    fs = result["fit_stats"]
    lines: list[str] = []

    lines.append("INTFIT: RELATIVE INTENSITY FIT")
    lines.append(f"Run name: {cfg.name}")
    lines.append(f"Spectrum: {cfg.spectrum_path}")
    lines.append("")
    lines.append("FIT QUALITY")
    lines.append(
        f" LINES USED={fs['n_obs']:5d}  NUMBER OF PARAMETERS={fs['n_par']:4d}  DOF={fs['dof']:5d}"
    )
    lines.append(
        f" RMS ERROR = {fs.get('unit_weight_rms', fs['rmse']):12.6f}   "
        f"MAX |OBS-CALC|/ERROR = {fs.get('max_abs_weighted_residual', np.nan):12.6f}"
    )
    lines.append(
        f" RAW AVG = {fs.get('raw_mean_residual', np.nan):13.6e}   "
        f"RAW RMS = {fs.get('raw_rmse', np.nan):13.6e}   "
        f"RAW MAX = {fs.get('raw_max_abs_residual', np.nan):13.6e}"
    )
    first_step = result.get("first_step_result")
    lines.append("")
    lines.append("OPTIONS")
    lines.append(f" extraction_mode        {cfg.fit_options.extraction_mode}")
    lines.append(f" extraction_window_MHz  {cfg.fit_options.extraction_window_MHz:.6f}")
    blend = result.get("blend")
    if blend:
        lines.append(
            f" blend_tolerance_MHz    {blend['tolerance_MHz']:.4f}  ({blend['n_transitions']} transitions -> {blend['n_peaks']} peaks, "
            f"{blend['n_blended_peaks']} of them blended" + ("; used for the line table only)" if blend.get("diagnostic_only") else ")")
        )
    lines.append(f" fit_in_log_space       {cfg.fit_options.fit_in_log_space}")
    lines.append(f" exact_T_scaling        {cfg.fit_options.use_exact_temperature_scaling}")
    lines.append(f" common_temperature     {cfg.fit_options.common_temperature}")
    lines.append(f" fit_mode               {cfg.fit_options.fit_mode}")
    lines.append("")
    lines.append("INPUT FILES AND PARSED METADATA")
    lines.append(" species                    lin cat int out egy var   mu_a/D      mu_b/D      mu_c/D      fit_mu")
    species_tables = species_tables or {}
    for sp in cfg.species:
        tab = species_tables.get(sp.name)
        attrs = tab.attrs if tab is not None else {}
        int_meta = attrs.get("int_metadata", {})
        dipoles = dict(attrs.get("dipoles", {}))
        q_table = attrs.get("partition_table")
        q_rows = len(q_table) if q_table is not None else 0
        fit_mu = ",".join(attrs.get("fit_dipole_components", []) or []) or "-"
        lines.append(
            f" {sp.name:<26} "
            f"{_yn(sp.lin_file):>3} {_yn(sp.cat_file):>3} {_yn(sp.int_file):>3} {_yn(sp.out_file):>3} {_yn(sp.egy_file):>3} {_yn(sp.var_file):>3} "
            f"{_fmt_mu(dipoles.get('a')):>10} {_fmt_mu(dipoles.get('b')):>10} {_fmt_mu(dipoles.get('c')):>10} {fit_mu}"
        )
        for warning in int_meta.get("warnings", []) if int_meta else []:
            lines.append(f"   WARNING: {warning}")
        if int_meta:
            title = int_meta.get("title", "")
            if title:
                lines.append(f"   .int title: {title}")
            if int_meta.get("exists", False):
                lines.append(
                    f"   .int controls: tag={int_meta.get('tag', np.nan):.0f} "
                    f"qrot_input={int_meta.get('qrot_input', np.nan):.6g} "
                    f"temperature_K={int_meta.get('temperature_K', np.nan):.6g} "
                    f"freq_max_GHz={int_meta.get('freq_max_GHz', np.nan):.6g}"
                )
        lines.append(
            f"   catalogue TEMP: {attrs.get('cat_temperature_K', float('nan')):.6g} K "
            f"({attrs.get('cat_temperature_source', 'unknown')}); intensities are scaled from here to Trot"
        )
        if q_rows:
            lines.append(f"   .out partition-function rows read: {q_rows}")
    lines.append("")
    if first_step:
        first_fs = first_step["fit_stats"]
        lines.append("STEP 1: ABUNDANCE AND TEMPERATURE FIT")
        lines.append(" dipole_model  from_cat")
        lines.append(f" success       {first_step['success']}")
        lines.append(f" status        {first_step['status']}")
        lines.append(f" message       {first_step['message']}")
        lines.append(f" nfev          {first_step['nfev']}")
        lines.append(f" n_obs         {first_fs['n_obs']}")
        lines.append(f" n_par         {first_fs['n_par']}")
        lines.append(f" rms           {first_fs.get('unit_weight_rms', first_fs['rmse']):.8e}  unit-weight, sqrt(RSS/dof)")
        lines.append(f" rms/obs       {first_fs.get('objective_rmse_nobs', first_fs['rmse']):.8e}  sqrt(RSS/n_obs)")
        lines.append(f" rms(raw)      {first_fs.get('raw_rmse', np.nan):.8e}  intensity units")
        lines.append("")
        lines.append("STEP 1 TEMPERATURES")
        if "Trot_fixed_K" in first_step:
            lines.append(f" GLOBAL_FIXED  {first_step['Trot_fixed_K']:.6f} K")
        else:
            for group, vals in first_step.get("temperature_groups", {}).items():
                if vals.get("is_fixed"):
                    lines.append(f" {group:<14} {vals['Trot_K']:.6f} K fixed")
                else:
                    lines.append(f" {group:<14} {vals['Trot_K']:.6f} +/- {vals['Trot_err_K']:.6f} K")
        lines.append("")
        lines.append("STEP 1 SPECIES RESULTS")
        lines.append(" species                    scale          rel_%     relmax    Trot/K")
        for name, vals in first_step.get("species", {}).items():
            lines.append(
                f" {name:<26} {vals['scale']:12.5e} "
                f"{100 * vals['relative_abundance']:8.3f} "
                f"{vals['population_relmax']:8.4f} {vals['Trot_K']:9.4f}"
            )
        lines.append("")
        lines.append("STEP 2: DIPOLE-ONLY FIT")
        lines.append(" fixed_from_step1 abundance/scale, relative populations, temperatures, FWHM")
        lines.append(f" dipole_weighting {fs.get('dipole_fit_residual_weighting', fs.get('residual_weighting', 'NA'))}")
        lines.append(
            f" dipole_lines candidate={fs.get('dipole_fit_candidate_lines', 0)} "
            f"used={fs.get('dipole_fit_used_lines', 0)} "
            f"rejected={fs.get('dipole_fit_rejected_lines', 0)} "
            f"reject_sigma={fs.get('dipole_fit_reject_sigma', np.nan):.3g}"
        )
    else:
        lines.append("OPTIMIZER")
    lines.append(f" success       {result['success']}")
    lines.append(f" status        {result['status']}")
    lines.append(f" message       {result['message']}")
    lines.append(f" nfev          {result['nfev']}")
    lines.append(f" optimality    {result['optimality']:.6e}")
    lines.append("")
    lines.append("STEP 2 FIT STATISTICS" if first_step else "GLOBAL FIT STATISTICS")
    lines.append(f" n_obs         {fs['n_obs']}")
    lines.append(f" n_par         {fs['n_par']}")
    lines.append(f" dof           {fs['dof']}")
    lines.append(f" rss           {fs['rss']:.8e}")
    lines.append(f" rms           {fs.get('unit_weight_rms', fs['rmse']):.8e}  unit-weight, sqrt(RSS/dof)")
    lines.append(f" rms/obs       {fs.get('objective_rmse_nobs', fs['rmse']):.8e}  sqrt(RSS/n_obs)")
    lines.append(f" rms(raw)      {fs.get('raw_rmse', np.nan):.8e}  intensity units")
    lines.append(f" rms(raw)*1e3  {1000.0 * fs.get('raw_rmse', np.nan):.6f}")
    lines.append(f" resid.std     {fs['residual_std']:.8e}")
    lines.append(f" sigma^2       {fs['sigma2']:.8e}")
    lines.append(f" cond(JTJ)     {fs['cond_JTJ']:.6e}")
    lines.append(f" intensity_unc {fs.get('intensity_uncertainty_fraction', np.nan):.4f}")
    if np.isfinite(fs.get("intensity_uncertainty_floor", np.nan)):
        lines.append(f" intensity_floor {fs.get('intensity_uncertainty_floor', np.nan):.8e}")
    lines.append(f" weighting     {fs.get('residual_weighting', 'fractional')}")
    if first_step and fs.get("dipole_fit_residual_model"):
        lines.append(f" step2_resid   {fs.get('dipole_fit_residual_model')}, sigma_log={fs.get('dipole_fit_sigma_log', np.nan):.6f}")
    lines.append(f" loss          {fs.get('loss', 'linear')}")
    lines.append(f" lineshape     {fs.get('lineshape_profile', 'NA')}, FWHM={fs.get('lineshape_fwhm_MHz', np.nan):.6f} MHz, fitted={fs.get('lineshape_fwhm_fitted', False)}")
    lines.append(f" n_mc          {fs.get('n_mc', 0)}")
    if np.isfinite(fs.get("residual_autocorr_factor", np.nan)):
        tau = fs["residual_autocorr_factor"]
        if fs.get("errors_scaled_by_autocorr", False):
            lines.append(f" resid.autocorr tau={tau:.2f}  (correlated spectrum points: covariance multiplied by tau, errors by sqrt(tau)={np.sqrt(tau):.2f})")
        else:
            lines.append(f" resid.autocorr tau={tau:.2f}  (correlated spectrum points: the errors below are NOT corrected, multiply them by sqrt(tau)={np.sqrt(tau):.2f})")
    lines.append("")
    lines.append("FIXED TEMPERATURES FROM STEP 1" if first_step else "TEMPERATURES")
    if "Trot_fixed_K" in result:
        lines.append(f" GLOBAL_FIXED  {result['Trot_fixed_K']:.6f} K")
    else:
        for group, vals in result.get("temperature_groups", {}).items():
            if vals.get("is_fixed"):
                lines.append(f" {group:<14} {vals['Trot_K']:.6f} K fixed")
            else:
                lines.append(f" {group:<14} {vals['Trot_K']:.6f} +/- {vals['Trot_err_K']:.6f} K")
    lines.append("")
    lines.append("STEP 2 DIPOLE SCALES" if first_step else "DIPOLE SCALES")
    if first_step and result.get("dipole_group_diagnostics"):
        lines.append(" component diagnostics")
        lines.append(" species                    comp status                    total valid used rejected med_exp/step1 prelim_mu final_mu red_chi2")
        for item in result["dipole_group_diagnostics"]:
            lines.append(
                f" {item['species']:<26} {item['component']:<4} {item['status']:<24} "
                f"{int(item['n_total']):5d} {int(item.get('n_valid', 0)):5d} {int(item['n_used']):4d} {int(item['n_rejected']):8d} "
                f"{item.get('median_ratio_step1', np.nan):12.4f} {item.get('prelim_mu_scale', np.nan):9.4f} "
                f"{item.get('final_mu_scale', np.nan):8.4f} {item.get('red_chi2', np.nan):8.3f}"
            )
        lines.append("")
    dipole_scales = result.get("dipole_scales", {})
    if dipole_scales:
        for key, vals in dipole_scales.items():
            species = vals.get("species", "*")
            comp = vals.get("component", "?")
            label = f"{species}:mu_{comp}" if species != "*" else f"GLOBAL:mu_{comp}"
            lines.append(f" {label:<22} scale={vals['scale']:.6f} +/- {vals.get('scale_err', np.nan):.6f}")
    else:
        lines.append(" none fitted")
    lines.append("")
    if result.get("dipole_rejected_lines"):
        lines.append("STEP 2 REJECTED DIPOLE-FIT LINES")
        lines.append(" species                    comp  lin_freq/MHz  exp            step1_calc     exp/step1  ratio_prelim  z_prelim  reason                         QN")
        for item in result["dipole_rejected_lines"]:
            lines.append(
                f" {item['species']:<26} {item['component']:<4} "
                f"{item['lin_freq_MHz']:12.5f} {item['exp_intensity']:13.6e} "
                f"{item['step1_model_intensity']:13.6e} {item['ratio_exp_step1']:9.4f} "
                f"{item.get('ratio_after_prelim', np.nan):13.4f} {item.get('z_after_prelim', np.nan):9.3f} "
                f"{item.get('reject_reason', 'rejected'):<30} "
                f"{item['QN_all']}"
            )
        lines.append("")
    lines.append("FINAL SPECIES RESULTS")
    rule = " " + "-" * 150
    sp_header = (
        " species                    label           n_lines  used  scale         +/-scale       rel_%    +/-rel_%   grp_%   +/-grp_%"
        "   relmax  +/-relmax    Trot/K    line_rmse  abc"
    )

    def _pct(value, width, decimals=3):
        return f"{'NA':>{width}}" if not np.isfinite(value) else f"{100 * value:{width}.{decimals}f}"

    groups_in_order = []
    for g in species_summary["group"] if "group" in species_summary.columns else []:
        g = g or "-"
        if g not in groups_in_order:
            groups_in_order.append(g)
    if "group" not in species_summary.columns:
        species_summary = species_summary.assign(group="-")
        groups_in_order = ["-"]
    for i_g, g in enumerate(groups_in_order):
        if i_g:
            lines.append("")
        sub = species_summary[species_summary["group"].fillna("").replace("", "-") == g]
        tot = sub["relative_abundance"].sum()
        lines.append(rule)
        lines.append(f" GROUP: {g}  ({len(sub)} species, sum of rel_% = {100 * tot:.3f})")
        lines.append(rule)
        lines.append(sp_header)
        for _, row in sub.iterrows():
            lines.append(
                f" {row['species']:<26} {str(row.get('label', row['species'])):<14} "
                f"{int(row['n_lines']):7d} {int(row['line_n_used']):5d} "
                f"{row['scale']:12.5e} {row.get('scale_err', np.nan):12.5e} {_pct(row['relative_abundance'], 8)} "
                f"{_pct(row.get('relative_abundance_err', np.nan), 9)} {_pct(row.get('group_fraction', np.nan), 7)} "
                f"{_pct(row.get('group_fraction_err', np.nan), 9)} "
                f"{row['population_relmax']:8.4f} {row.get('population_relmax_err', np.nan):9.4f} {row['Trot_K']:9.4f} {row['line_rmse']:10.4f} "
                f"{int(row.get('n_a_type', 0))}/{int(row.get('n_b_type', 0))}/{int(row.get('n_c_type', 0))}"
            )
        lines.append(rule)

    conf = result.get("conformer_analysis") or {}
    if conf.get("rows"):
        per_group = bool(conf.get("groups"))
        lines.append("")
        lines.append("CONFORMER POPULATIONS (frozen Boltzmann at Tconf vs fitted abundances)")
        if per_group:
            for g, ginfo in conf["groups"].items():
                for ref, info in ginfo["fits"].items():
                    err = info.get("Tconf_err_K", np.nan)
                    err_txt = f" +/- {err:.1f}" if np.isfinite(err) else ""
                    lines.append(f" {g}: {ref}: Tconf = {info['Tconf_K']:.1f}{err_txt} K  ({info['source']}, {info['n_species']} species)")
        else:
            for ref, info in conf["fits"].items():
                err = info.get("Tconf_err_K", np.nan)
                err_txt = f" +/- {err:.1f}" if np.isfinite(err) else ""
                lines.append(f" {ref}: Tconf = {info['Tconf_K']:.1f}{err_txt} K  ({info['source']}, {info['n_species']} species)")
        if conf.get("excluded"):
            lines.append(f" excluded_from_Tconf_fit: {', '.join(conf['excluded'])}")
        rule_c = " " + "-" * 111
        from .conformers import energy_unit_to_cm1
        unit = str(conf.get("unit", "cm-1"))
        e_factor = energy_unit_to_cm1(unit)
        e_dec = {"hartree": 7, "ev": 5}.get(unit.strip().lower(), 2)
        e_head = f"E/{unit}"
        c_header = f" {'species':<26} {'label':<14} {e_head:>9} {'degen':>8} {'expected_%':>10} {'fitted_%':>9} {'err_%':>7} {'fit/exp':>8}"
        blocks = {}
        for item in conf["rows"]:
            key = (str(item.get("group", "") or "-"), item["reference"], float(item["Tconf_K"]), str(item.get("Tconf_source", "")))
            blocks.setdefault(key, []).append(item)
        for i_b, ((g, ref, tconf, source), items) in enumerate(blocks.items()):
            lines.append("")
            exp = np.array([100 * it["expected_fraction"] for it in items])
            fit = np.array([100 * it["fitted_fraction"] for it in items])
            ratio = np.array([it["fitted_over_expected"] for it in items])
            rms = float(np.sqrt(np.mean((fit - exp) ** 2)))
            ok = np.isfinite(ratio) & (ratio > 0)
            rms_ln = float(np.sqrt(np.mean(np.log(ratio[ok]) ** 2))) if ok.any() else float("nan")
            lines.append(rule_c)
            lines.append(f" GROUP: {g}   ref: {ref}   Tconf = {tconf:.1f} K  ({source or 'n/a'})   {len(items)} species")
            lines.append(rule_c)
            lines.append(c_header)
            for it in items:
                lines.append(
                    f" {it['species']:<26} {str(it.get('label', it['species'])):<14} {it['energy_cm-1'] / e_factor:9.{e_dec}f} {it['degeneracy']:8.3f} "
                    f"{100 * it['expected_fraction']:10.3f} {100 * it['fitted_fraction']:9.3f} "
                    f"{100 * it['fitted_fraction_err']:7.3f} {it['fitted_over_expected']:8.3f}"
                )
            lines.append(rule_c)
            lines.append(f" RMS for {g} {ref} at Tconf {tconf:.1f} K:   RMS(fitted_% - expected_%) = {rms:.3f} %-points   RMS(ln fit/exp) = {rms_ln:.3f}")
            lines.append(rule_c)

    lines.append("")
    lines.append("WORST LINES BY ABSOLUTE RESIDUAL")
    if line_diag.empty:
        lines.append(" none")
    else:
        ranked = line_diag[np.isfinite(line_diag["abs_residual"])].sort_values("abs_residual", ascending=False)
        if "cluster" in ranked.columns:  # blended peaks are listed once, with the summed model
            ranked = pd.concat([ranked[ranked["cluster"] < 0], ranked[ranked["cluster"] >= 0].drop_duplicates("cluster")]).sort_values(
                "abs_residual", ascending=False)
        worst = ranked.head(20)
        lines.append(" species                    lin_freq/MHz  exp            calc           residual       QN")
        for _, row in worst.iterrows():
            has_peak = "cluster" in row and row["cluster"] >= 0 and np.isfinite(row["cluster_model"])
            blended = has_peak and row["n_in_cluster"] > 1
            exp_v = row["cluster_exp"] if has_peak else row["exp_intensity"]
            calc_v = row["cluster_model"] if has_peak else row["model_intensity"]
            lines.append(
                f" {row['species']:<26} {row['lin_freq_MHz']:12.5f} "
                f"{exp_v:13.6e} {calc_v:13.6e} "
                f"{row['residual']:13.6e} {row['QN_all']}" + (f"  [+{int(row['n_in_cluster']) - 1} blended]" if blended else "")
            )

    if result.get("warnings"):
        lines.append("")
        lines.append("WARNINGS")
        for warning in result["warnings"]:
            lines.append(f" - {warning}")

    path.write_text("\n".join(lines) + "\n")


def print_console_summary(cfg: RunConfig, result: dict, species_summary: pd.DataFrame, report_path: str | Path) -> None:
    fs = result["fit_stats"]
    print("\n=== INTFIT SUMMARY ===")
    print(f"run       : {cfg.name}")
    print(f"success   : {result['success']} ({result['message']})")
    print(f"LINES USED={fs['n_obs']:5d}  NUMBER OF PARAMETERS={fs['n_par']:4d}  DOF={fs['dof']:5d}")
    print(
        f"RMS ERROR = {fs.get('unit_weight_rms', fs['rmse']):12.6f}   "
        f"MAX |OBS-CALC|/ERROR = {fs.get('max_abs_weighted_residual', np.nan):12.6f}"
    )
    print(
        f"RAW AVG = {fs.get('raw_mean_residual', np.nan):13.6e}   "
        f"RAW RMS = {fs.get('raw_rmse', np.nan):13.6e}   "
        f"RAW MAX = {fs.get('raw_max_abs_residual', np.nan):13.6e}"
    )
    print(f"report    : {report_path}")
    print("\nSpecies:")
    cols = ["species"] + (["label"] if "label" in species_summary.columns and (species_summary["label"] != species_summary["species"]).any() else []) + ["n_lines", "line_n_used", "relative_abundance", "population_relmax", "Trot_K", "line_rmse"]
    print(species_summary[cols].to_string(index=False))
