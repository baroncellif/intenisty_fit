from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FitOptions, SpeciesConfig
from .constants import DEFAULT_CAT_TEMPERATURE_K
from .model import intensity_from_lgint, intensity_at_T_approx, intensity_at_T_exact
from .pickett_io import apply_filters, classify_dipole_component, match_lin_to_cat_exact, read_cat, read_lin
from .pickett_meta import read_int_metadata, read_out_partition_function
from .spectrum import extract_line_intensity


def build_species_prediction_table(species_cfg: SpeciesConfig, fit_options: FitOptions) -> pd.DataFrame:
    lin_df = read_lin(species_cfg.lin_file)
    cat_df = read_cat(species_cfg.cat_file)

    if lin_df.empty:
        raise ValueError(f"{species_cfg.name}: .lin parser returned 0 rows")
    if cat_df.empty:
        raise ValueError(f"{species_cfg.name}: .cat parser returned 0 rows")

    lin_df = apply_filters(
        lin_df.rename(columns={"lin_freq_MHz": "freq_MHz"}),
        species_cfg.filters,
    ).rename(columns={"freq_MHz": "lin_freq_MHz"})

    if lin_df.empty:
        raise ValueError(f"{species_cfg.name}: no .lin rows left after filtering")

    lin_df["dipole_component"] = classify_dipole_component(lin_df)
    lin_component_counts = lin_df["dipole_component"].value_counts().to_dict()

    merged = match_lin_to_cat_exact(lin_df, cat_df)
    for col in ["QN_all", "QN_up", "QN_lo", "nqn", "Jp", "Kap", "Kcp", "Vp", "F1p", "F2p", "J", "Ka", "Kc", "V", "F1", "F2"]:
        col_lin = f"{col}_lin"
        if col_lin in merged.columns:
            merged[col] = merged[col_lin]

    missing = merged["match_mode"] == "MISSING_QN"
    missing_component_counts = (
        merged.loc[missing, "dipole_component"].value_counts().to_dict()
        if "dipole_component" in merged.columns
        else {}
    )
    merged = merged.loc[~missing].copy()
    if merged.empty:
        raise ValueError(f"{species_cfg.name}: no LIN transition has an exact QN match in CAT")

    merged["delta_MHz"] = np.abs(merged["cat_freq_MHz"] - merged["lin_freq_MHz"])

    if species_cfg.filters is not None and "delta_max" in species_cfg.filters:
        merged = merged[merged["delta_MHz"] <= species_cfg.filters["delta_max"]].copy()
    if merged.empty:
        raise ValueError(f"{species_cfg.name}: no transitions left after delta_max filtering")

    merged["I_cat"] = intensity_from_lgint(merged["lgint"].to_numpy())
    merged["dipole_component"] = classify_dipole_component(merged)

    int_meta = read_int_metadata(species_cfg.int_file)
    # LGINT is SPCAT's intensity at the TEMP of the .int that produced the
    # .cat; every temperature scaling starts from there.
    cat_temperature = int_meta.get("temperature_K")
    if cat_temperature is not None and np.isfinite(cat_temperature) and cat_temperature > 0:
        cat_temperature_source = ".int"
    else:
        cat_temperature = DEFAULT_CAT_TEMPERATURE_K
        cat_temperature_source = "default (no .int TEMP)"
    merged["cat_temperature_K"] = float(cat_temperature)
    q_table = read_out_partition_function(species_cfg.out_file)
    dipoles = dict(int_meta.get("dipoles", {}))

    def dipole_for_component(component: str) -> float:
        val = dipoles.get(str(component), np.nan)
        return float(val) if val is not None else np.nan

    merged["dipole_used_D"] = merged["dipole_component"].map(dipole_for_component).astype(float)
    mu2 = merged["dipole_used_D"].to_numpy(dtype=float) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["I_cat_unit_dipole"] = np.where(mu2 > 0, merged["I_cat"].to_numpy(dtype=float) / mu2, np.nan)

    merged["species"] = species_cfg.name
    merged.attrs["fixed_scale"] = species_cfg.fixed_scale
    merged.attrs["fit_dipole_components"] = [str(c).lower() for c in (species_cfg.fit_dipole_components or [])]
    merged.attrs["n_lin_read"] = len(lin_df)
    merged.attrs["n_cat_read"] = len(cat_df)
    merged.attrs["n_missing_qn"] = int(missing.sum())
    merged.attrs["lin_filtered_component_counts"] = {str(k): int(v) for k, v in lin_component_counts.items()}
    merged.attrs["missing_qn_component_counts"] = {str(k): int(v) for k, v in missing_component_counts.items()}
    merged.attrs["int_metadata"] = int_meta
    merged.attrs["cat_temperature_K"] = float(cat_temperature)
    merged.attrs["cat_temperature_source"] = cat_temperature_source
    merged.attrs["partition_table"] = q_table
    merged.attrs["dipoles"] = dipoles
    merged.attrs["delta_E_cm1"] = species_cfg.delta_E_cm1
    merged.attrs["conformer_degeneracy"] = species_cfg.conformer_degeneracy
    merged.attrs["dEe"] = species_cfg.dEe
    merged.attrs["dE0"] = species_cfg.dE0
    merged.attrs["group"] = species_cfg.group
    merged.attrs["label"] = species_cfg.label or species_cfg.name
    return merged.reset_index(drop=True)


def build_species_tables(species: list[SpeciesConfig], fit_options: FitOptions) -> dict[str, pd.DataFrame]:
    return {sp.name: build_species_prediction_table(sp, fit_options) for sp in species}


def build_observation_table(
    species_tables: dict[str, pd.DataFrame],
    spectrum_df: pd.DataFrame,
    fit_options: FitOptions,
) -> pd.DataFrame:
    rows = []

    for name, tab in species_tables.items():
        exp_I = extract_line_intensity(
            spectrum_df,
            tab["lin_freq_MHz"].to_numpy(dtype=float),
            mode=fit_options.extraction_mode,
            window_MHz=fit_options.extraction_window_MHz,
        )

        for i in range(len(tab)):
            rows.append({
                "species": name,
                "lin_freq_MHz": float(tab.iloc[i]["lin_freq_MHz"]),
                "cat_freq_MHz": float(tab.iloc[i]["cat_freq_MHz"]),
                "delta_MHz": float(tab.iloc[i]["delta_MHz"]),
                "I_cat": float(tab.iloc[i]["I_cat"]),
                "I_cat_unit_dipole": float(tab.iloc[i].get("I_cat_unit_dipole", np.nan)),
                "dipole_component": tab.iloc[i].get("dipole_component", "unknown"),
                "dipole_used_D": float(tab.iloc[i].get("dipole_used_D", np.nan)),
                "elo_cm-1": float(tab.iloc[i]["elo_cm-1"]),
                "dr": float(tab.iloc[i]["dr"]),
                "cat_temperature_K": float(tab.iloc[i]["cat_temperature_K"]),
                "lgint": float(tab.iloc[i]["lgint"]),
                "exp_intensity": float(exp_I[i]) if np.isfinite(exp_I[i]) else np.nan,
                "QN_all": tab.iloc[i]["QN_all"],
            })

    obs = pd.DataFrame(rows)
    obs = obs[np.isfinite(obs["exp_intensity"])].reset_index(drop=True)
    if obs.empty:
        raise ValueError("No finite experimental intensities were extracted from the spectrum")
    return obs


def build_line_diagnostic_table(species_tables, spectrum_df, result, fit_options, q_functions=None) -> pd.DataFrame:
    q_functions = q_functions or {}
    rows = []
    cl = {}
    obs_tab = result.get("observation_table")
    if obs_tab is not None and "cluster_model" in obs_tab.columns:
        for r in obs_tab.itertuples():
            cl[(r.species, round(float(r.lin_freq_MHz), 6), str(r.QN_all))] = (int(r.cluster), int(r.n_in_cluster), float(r.cluster_exp), float(r.cluster_model))

    for name, tab in species_tables.items():
        Tnow = result["species"][name]["Trot_K"]
        scale = result["species"][name]["scale"]
        exp_I = extract_line_intensity(
            spectrum_df,
            tab["lin_freq_MHz"].to_numpy(dtype=float),
            mode=fit_options.extraction_mode,
            window_MHz=fit_options.extraction_window_MHz,
        )
        I_cat = tab["I_cat"].to_numpy(dtype=float)
        if fit_options.dipole_model == "component_scaled" and result.get("dipole_scales"):
            unit = tab["I_cat_unit_dipole"].to_numpy(dtype=float)
            mu_used = tab["dipole_used_D"].to_numpy(dtype=float)
            comps = tab["dipole_component"].astype(str).to_numpy()
            I_cat = I_cat.copy()
            valid = np.isfinite(unit) & np.isfinite(mu_used)
            I_cat[valid] = unit[valid] * mu_used[valid] ** 2
            for _key, vals in result["dipole_scales"].items():
                target_species = vals.get("species", "*")
                component = vals.get("component")
                if target_species not in {"*", name}:
                    continue
                scale_mu = vals["scale"]
                cmask = valid & (comps == component)
                I_cat[cmask] = unit[cmask] * (mu_used[cmask] * scale_mu) ** 2
        Elo = tab["elo_cm-1"].to_numpy(dtype=float)
        DR = tab["dr"].to_numpy(dtype=float)
        nu = tab["cat_freq_MHz"].to_numpy(dtype=float)
        Tcat = tab["cat_temperature_K"].to_numpy(dtype=float)

        if fit_options.use_exact_temperature_scaling and name in q_functions:
            Icat_T = intensity_at_T_exact(I_cat, Elo, nu, Tnow, Tcat, q_functions[name])
        else:
            Icat_T = intensity_at_T_approx(I_cat, Elo, DR, Tnow, Tcat)

        model = scale * Icat_T
        residual = exp_I - model

        for i in range(len(tab)):
            blended = cl.get((name, round(float(tab.iloc[i]["lin_freq_MHz"]), 6), str(tab.iloc[i]["QN_all"])))
            if blended is not None and np.isfinite(blended[2]) and np.isfinite(blended[3]):
                res_i = blended[2] - blended[3]
            else:
                res_i = float(residual[i]) if np.isfinite(residual[i]) else np.nan
            rows.append({
                "species": name,
                "temperature_group": result["species"][name].get("temperature_group", "NA"),
                "Trot_K": Tnow,
                "lin_freq_MHz": float(tab.iloc[i]["lin_freq_MHz"]),
                "cat_freq_MHz": float(tab.iloc[i]["cat_freq_MHz"]),
                "delta_MHz": float(tab.iloc[i]["delta_MHz"]),
                "exp_intensity": float(exp_I[i]) if np.isfinite(exp_I[i]) else np.nan,
                "model_intensity": float(model[i]),
                "cluster": blended[0] if blended else -1,
                "n_in_cluster": blended[1] if blended else 1,
                "cluster_exp": blended[2] if blended else np.nan,
                "cluster_model": blended[3] if blended else np.nan,
                "residual": res_i,
                "abs_residual": abs(res_i) if np.isfinite(res_i) else np.nan,
                "ELO_cm-1": float(tab.iloc[i]["elo_cm-1"]),
                "lgint": float(tab.iloc[i]["lgint"]),
                "I_cat": float(tab.iloc[i]["I_cat"]),
                "I_cat_unit_dipole": float(tab.iloc[i].get("I_cat_unit_dipole", np.nan)),
                "dipole_component": tab.iloc[i].get("dipole_component", "unknown"),
                "dipole_used_D": float(tab.iloc[i].get("dipole_used_D", np.nan)),
                "Jp": int(tab.iloc[i]["Jp"]),
                "Kap": int(tab.iloc[i]["Kap"]),
                "Kcp": int(tab.iloc[i]["Kcp"]),
                "J": int(tab.iloc[i]["J"]),
                "Ka": int(tab.iloc[i]["Ka"]),
                "Kc": int(tab.iloc[i]["Kc"]),
                "QN_all": str(tab.iloc[i]["QN_all"]),
            })

    out = pd.DataFrame(rows)
    return out.sort_values(["species", "lin_freq_MHz"]).reset_index(drop=True) if not out.empty else out


def _nan_if_none(value) -> float:
    return np.nan if value is None else float(value)


def residual_metrics(residuals) -> dict[str, float]:
    rr = np.asarray(residuals, dtype=float)
    rr = rr[np.isfinite(rr)]
    if rr.size == 0:
        return {"line_n_used": 0, "line_rmse": np.nan, "line_mae": np.nan, "line_mean_residual": np.nan}
    return {
        "line_n_used": int(rr.size),
        "line_rmse": float(np.sqrt(np.mean(rr ** 2)) * 1e3),
        "line_mae": float(np.mean(np.abs(rr))),
        "line_mean_residual": float(np.mean(rr)),
    }


def build_species_diagnostic_summary(species_tables, result, line_diag_df=None) -> pd.DataFrame:
    rows = []
    for name, tab in species_tables.items():
        vals = result["species"][name]
        if line_diag_df is not None and not line_diag_df.empty:
            sub = line_diag_df[line_diag_df["species"] == name]
            metrics = residual_metrics(sub["residual"].to_numpy(dtype=float))
        else:
            metrics = residual_metrics([])

        rows.append({
            "species": name,
            "label": tab.attrs.get("label") or name,
            "group": tab.attrs.get("group") or "",
            "temperature_group": vals.get("temperature_group", "NA"),
            "n_lines": int(len(tab)),
            **metrics,
            "freq_min_MHz": float(tab["cat_freq_MHz"].min()),
            "freq_max_MHz": float(tab["cat_freq_MHz"].max()),
            "elo_min_cm-1": float(tab["elo_cm-1"].min()),
            "elo_max_cm-1": float(tab["elo_cm-1"].max()),
            "scale": vals["scale"],
            "scale_err": vals.get("scale_err", np.nan),
            "relative_abundance_err": vals.get("relative_abundance_err", np.nan),
            "relative_abundance_mc_rms": vals.get("relative_abundance_mc_rms", np.nan),
            "population_relmax_err": vals.get("population_relmax_err", np.nan),
            "relative_abundance": vals["relative_abundance"],
            "population_relmax": vals["population_relmax"],
            "Trot_K": vals["Trot_K"],
            "Trot_err_K": vals.get("Trot_err_K", np.nan),
            "n_a_type": int((tab.get("dipole_component", pd.Series(dtype=str)) == "a").sum()),
            "n_b_type": int((tab.get("dipole_component", pd.Series(dtype=str)) == "b").sum()),
            "n_c_type": int((tab.get("dipole_component", pd.Series(dtype=str)) == "c").sum()),
            "delta_E_cm-1": tab.attrs.get("delta_E_cm1", np.nan),
            "dEe": _nan_if_none(tab.attrs.get("dEe")),
            "dE0": _nan_if_none(tab.attrs.get("dE0")),
            "conformer_degeneracy": tab.attrs.get("conformer_degeneracy", np.nan),
            "fit_global_rmse": result.get("fit_stats", {}).get("rmse", np.nan),
            "fit_global_rss": result.get("fit_stats", {}).get("rss", np.nan),
        })
    return pd.DataFrame(rows)
