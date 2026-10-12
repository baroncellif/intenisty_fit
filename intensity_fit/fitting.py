from __future__ import annotations

import copy
from dataclasses import replace
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from .config import FitOptions
from .lineshape import lineshape_options, render_profiles
from .model import intensity_at_T_approx, intensity_at_T_exact
from .tables import build_observation_table


def _intensity_sigma(y: np.ndarray, fit_options: FitOptions, residual_weighting: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if residual_weighting == "none":
        return np.ones_like(y, dtype=float)

    frac = float(fit_options.intensity_uncertainty_fraction)
    fractional = np.abs(y) * frac

    if residual_weighting == "fractional":
        return np.maximum(fractional, np.finfo(float).tiny)

    if residual_weighting == "hybrid":
        floor = fit_options.intensity_uncertainty_floor
        if floor is None:
            floor = 0.0
        floor = float(floor)
        if floor < 0.0:
            raise ValueError("intensity_uncertainty_floor must be >= 0")
        return np.maximum(np.sqrt(fractional ** 2 + floor ** 2), np.finfo(float).tiny)

    raise ValueError("residual_weighting must be 'none', 'fractional', or 'hybrid'")


def build_group_temperature_structure(species_names: list[str], fit_options: FitOptions) -> dict:
    fixed_group_temperatures = dict(fit_options.fixed_group_temperatures or {})

    if fit_options.fixed_temperature_K is not None:
        return {
            "mode": "fixed_global",
            "group_names": [],
            "species_to_group": None,
            "fixed_group_temperatures": {},
            "fitted_group_names": [],
        }

    if fit_options.species_temperature_groups is None:
        if fit_options.common_temperature:
            species_to_group = {name: "ALL" for name in species_names}
            group_names = ["ALL"]
        else:
            species_to_group = {name: name for name in species_names}
            group_names = species_names[:]
    else:
        species_to_group = dict(fit_options.species_temperature_groups)
        missing = [name for name in species_names if name not in species_to_group]
        if missing:
            raise ValueError(f"Missing temperature group assignment for species: {missing}")
        group_names = sorted(set(species_to_group.values()))

    unknown_fixed = [g for g in fixed_group_temperatures if g not in group_names]
    if unknown_fixed:
        raise ValueError(f"Fixed temperatures provided for unknown groups: {unknown_fixed}")

    return {
        "mode": "grouped",
        "group_names": group_names,
        "species_to_group": species_to_group,
        "fixed_group_temperatures": fixed_group_temperatures,
        "fitted_group_names": [g for g in group_names if g not in fixed_group_temperatures],
    }


def evaluate_species_temperatures(species_names, fit_options, temp_structure, p, n_fitted_species):
    if fit_options.fixed_temperature_K is not None:
        T = float(fit_options.fixed_temperature_K)
        return {name: T for name in species_names}, {"GLOBAL_FIXED": T}

    group_to_T: dict[str, float] = {}
    for g, T in temp_structure["fixed_group_temperatures"].items():
        group_to_T[g] = float(T)

    temp_params = p[n_fitted_species:]
    for i, g in enumerate(temp_structure["fitted_group_names"]):
        group_to_T[g] = float(np.exp(temp_params[i]))

    species_to_group = temp_structure["species_to_group"]
    return {name: group_to_T[species_to_group[name]] for name in species_names}, group_to_T


def fit_mixture(
    species_tables: dict,
    spectrum_df,
    fit_options: FitOptions,
    q_functions: dict[str, Callable[[float], float]] | None = None,
) -> dict:
    q_functions = q_functions or {}

    species_names = list(species_tables.keys())
    fixed_scale_map = {name: species_tables[name].attrs.get("fixed_scale", None) for name in species_names}
    fitted_species = [name for name in species_names if fixed_scale_map[name] is None]
    fixed_species = [name for name in species_names if fixed_scale_map[name] is not None]
    n_fitted_species = len(fitted_species)
    dipole_targets = _build_dipole_targets(species_tables, fit_options)
    if fit_options.dipole_model != "component_scaled":
        dipole_targets = []
    if dipole_targets and fit_options.dipole_fit_second_step:
        return _fit_mixture_dipole_second_step(species_tables, spectrum_df, fit_options, q_functions, dipole_targets)
    n_dipole = len(dipole_targets)
    fit_mode = str(fit_options.fit_mode).lower()
    lopts = lineshape_options(fit_options)
    fit_fwhm = fit_mode == "spectrum" and lopts["fit_fwhm"]
    n_fwhm = 1 if fit_fwhm else 0

    obs = build_observation_table(species_tables, spectrum_df, fit_options)
    y_exp_line = obs["exp_intensity"].to_numpy(dtype=float)
    residual_weighting = str(fit_options.residual_weighting).lower()
    sigma_line = _intensity_sigma(y_exp_line, fit_options, residual_weighting)
    spec_x = spec_y = sigma_spec = None
    if fit_mode == "spectrum":
        spec_x, spec_y = _spectrum_points_for_windows(spectrum_df, obs["lin_freq_MHz"].to_numpy(dtype=float), lopts["window_MHz"])
        sigma_spec = _intensity_sigma(spec_y, fit_options, residual_weighting)
        y_exp = spec_y
    else:
        y_exp = y_exp_line
    species_index = {name: i for i, name in enumerate(species_names)}
    obs_species_idx = obs["species"].map(species_index).to_numpy(dtype=int)

    temp_structure = build_group_temperature_structure(species_names, fit_options)
    fitted_group_names = temp_structure["fitted_group_names"]

    if fit_options.fixed_temperature_K is not None:
        p0 = np.r_[np.zeros(n_fitted_species + n_dipole), np.full(n_fwhm, np.log(lopts["fwhm_MHz"]))]
        lb = np.r_[np.full(n_fitted_species + n_dipole, -50.0), np.full(n_fwhm, np.log(lopts["fwhm_bounds_MHz"][0]))]
        ub = np.r_[np.full(n_fitted_species + n_dipole, 50.0), np.full(n_fwhm, np.log(lopts["fwhm_bounds_MHz"][1]))]
    else:
        n_fit_groups = len(fitted_group_names)
        p0 = np.r_[
            np.zeros(n_fitted_species),
            np.zeros(n_dipole),
            np.full(n_fwhm, np.log(lopts["fwhm_MHz"])),
            np.full(n_fit_groups, np.log(fit_options.temp_init_K)),
        ]
        lb = np.r_[
            np.full(n_fitted_species, -50.0),
            np.full(n_dipole, np.log(0.05)),
            np.full(n_fwhm, np.log(lopts["fwhm_bounds_MHz"][0])),
            np.full(n_fit_groups, np.log(fit_options.temp_min_K)),
        ]
        ub = np.r_[
            np.full(n_fitted_species, 50.0),
            np.full(n_dipole, np.log(20.0)),
            np.full(n_fwhm, np.log(lopts["fwhm_bounds_MHz"][1])),
            np.full(n_fit_groups, np.log(fit_options.temp_max_K)),
        ]

    I_cat_all = obs["I_cat"].to_numpy(dtype=float)
    I_cat_unit_all = obs.get("I_cat_unit_dipole", obs["I_cat"]).to_numpy(dtype=float)
    component_all = obs.get("dipole_component", "").astype(str).to_numpy()
    dipole_used_all = obs.get("dipole_used_D", np.nan).to_numpy(dtype=float)
    Elo_all = obs["elo_cm-1"].to_numpy(dtype=float)
    DR_all = obs["dr"].to_numpy(dtype=float)
    nu_all = obs["cat_freq_MHz"].to_numpy(dtype=float)
    Tcat_all = obs["cat_temperature_K"].to_numpy(dtype=float)

    def evaluate_model(p: np.ndarray):
        A_map: dict[str, float] = {}
        for i, name in enumerate(fitted_species):
            A_map[name] = float(np.exp(p[i]))
        for name in fixed_species:
            A_map[name] = float(fixed_scale_map[name])

        dipole_offset = n_fitted_species
        fwhm_offset = n_fitted_species + n_dipole
        temp_offset = n_fitted_species + n_dipole + n_fwhm
        p_for_temp = np.r_[p[:n_fitted_species], p[temp_offset:]]
        Ts, group_to_T = evaluate_species_temperatures(
            species_names, fit_options, temp_structure, p_for_temp, n_fitted_species
        )
        dipole_scale = {
            target: float(np.exp(p[dipole_offset + i]))
            for i, target in enumerate(dipole_targets)
        }

        y_model = np.zeros_like(y_exp_line, dtype=float)
        for i, name in enumerate(species_names):
            mask = obs_species_idx == i
            if not np.any(mask):
                continue
            Tnow = Ts[name]
            base_I_cat = I_cat_all[mask]
            if fit_options.dipole_model == "component_scaled":
                unit = I_cat_unit_all[mask]
                comps = component_all[mask]
                mu_used = dipole_used_all[mask]
                base_I_cat = base_I_cat.copy()
                valid_unit = np.isfinite(unit)
                base_I_cat[valid_unit] = unit[valid_unit] * mu_used[valid_unit] ** 2
                for target, scale in dipole_scale.items():
                    target_species, component = target
                    if target_species not in {"*", name}:
                        continue
                    cmask = valid_unit & (comps == component)
                    base_I_cat[cmask] = unit[cmask] * (mu_used[cmask] * scale) ** 2

            if fit_options.use_exact_temperature_scaling and name in q_functions:
                pred = intensity_at_T_exact(base_I_cat, Elo_all[mask], nu_all[mask], Tnow, Tcat_all[mask], q_functions[name])
            else:
                pred = intensity_at_T_approx(base_I_cat, Elo_all[mask], DR_all[mask], Tnow, Tcat_all[mask])
            y_model[mask] = A_map[name] * pred
        fwhm = float(np.exp(p[fwhm_offset])) if fit_fwhm else float(lopts["fwhm_MHz"])
        return y_model, A_map, Ts, group_to_T, dipole_scale, fwhm

    def residuals(p: np.ndarray) -> np.ndarray:
        y_model_line, _A, _Ts, _groups, _dipoles, fwhm = evaluate_model(p)
        if fit_mode == "spectrum":
            y_model = render_profiles(
                spec_x,
                obs["lin_freq_MHz"].to_numpy(dtype=float),
                y_model_line,
                fwhm=fwhm,
                profile=lopts["profile"],
                normalize=lopts["normalize"],
            )
            return (spec_y - y_model) / sigma_spec
        y_model = y_model_line
        if fit_options.fit_in_log_space:
            eps = 1e-30
            sigma_log = max(np.log10(1.0 + float(fit_options.intensity_uncertainty_fraction)), 1e-12)
            return (np.log10(y_exp_line + eps) - np.log10(y_model + eps)) / sigma_log
        return (y_exp_line - y_model) / sigma_line

    res = least_squares(
        residuals,
        p0,
        bounds=(lb, ub),
        method="trf",
        loss=fit_options.loss,
        f_scale=fit_options.loss_f_scale,
    )
    y_model_line, A_map, Ts_final, group_to_T_final, dipole_scale_final, fwhm_final = evaluate_model(res.x)
    if fit_mode == "spectrum":
        y_model = render_profiles(
            spec_x,
            obs["lin_freq_MHz"].to_numpy(dtype=float),
            y_model_line,
            fwhm=fwhm_final,
            profile=lopts["profile"],
            normalize=lopts["normalize"],
        )
    else:
        y_model = y_model_line
    r = np.asarray(res.fun, dtype=float)
    raw_residual = y_exp - y_model
    raw_line_residual = y_exp_line - y_model_line

    n_obs = int(r.size)
    n_par = int(res.x.size)
    dof = max(n_obs - n_par, 1)
    rss = float(np.sum(r ** 2))
    objective_rmse_nobs = float(np.sqrt(np.mean(r ** 2)))
    sigma2 = float(rss / dof)
    unit_weight_rms = float(np.sqrt(sigma2))
    residual_std = float(np.std(r, ddof=1)) if n_obs > 1 else np.nan
    max_abs_weighted_residual = float(np.max(np.abs(r))) if n_obs else np.nan
    mean_weighted_residual = float(np.mean(r)) if n_obs else np.nan

    cov = None
    perr = None
    cond_JTJ = np.nan
    try:
        J = np.asarray(res.jac, dtype=float)
        JTJ = J.T @ J
        cond_JTJ = float(np.linalg.cond(JTJ))
        cov = sigma2 * np.linalg.pinv(JTJ)
        perr = np.sqrt(np.diag(cov))
    except Exception:
        pass

    A = np.array([A_map[name] for name in species_names], dtype=float)
    rel_sum = A / np.sum(A) if np.sum(A) > 0 else np.full_like(A, np.nan)
    rel_max = A / np.max(A) if np.max(A) > 0 else np.full_like(A, np.nan)

    out = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "cost": float(res.cost),
        "nfev": int(res.nfev),
        "optimality": float(res.optimality),
        "x0": p0.tolist(),
        "x_final": res.x.tolist(),
        "residual_vector": r.tolist(),
        "raw_residual_vector": raw_residual.tolist(),
        "y_exp": y_exp.tolist(),
        "y_model": y_model.tolist(),
        "y_exp_line": y_exp_line.tolist(),
        "y_model_line": y_model_line.tolist(),
        "observation_table": obs,
        "fit_stats": {
            "n_obs": n_obs,
            "n_par": n_par,
            "dof": int(dof),
            "rss": rss,
            "rmse": unit_weight_rms,
            "unit_weight_rms": unit_weight_rms,
            "objective_rmse_nobs": objective_rmse_nobs,
            "raw_rmse": float(np.sqrt(np.mean(raw_residual ** 2))),
            "residual_std": residual_std,
            "max_abs_weighted_residual": max_abs_weighted_residual,
            "mean_weighted_residual": mean_weighted_residual,
            "raw_mean_residual": float(np.mean(raw_residual)),
            "raw_max_abs_residual": float(np.max(np.abs(raw_residual))),
            "sigma2": sigma2,
            "cond_JTJ": cond_JTJ,
            "intensity_uncertainty_fraction": float(fit_options.intensity_uncertainty_fraction),
            "intensity_uncertainty_floor": (
                float(fit_options.intensity_uncertainty_floor)
                if fit_options.intensity_uncertainty_floor is not None
                else np.nan
            ),
            "residual_weighting": residual_weighting,
            "loss": fit_options.loss,
            "fit_mode": fit_mode,
            "lineshape_profile": lopts["profile"],
            "lineshape_fwhm_MHz": fwhm_final,
            "lineshape_fwhm_fitted": fit_fwhm,
            "lineshape_normalize": lopts["normalize"],
            "line_raw_rmse": float(np.sqrt(np.mean(raw_line_residual ** 2))),
        },
        "species": {},
        "dipole_scales": _format_dipole_scales(dipole_scale_final),
    }

    if cov is not None:
        out["covariance_matrix"] = cov.tolist()
        out["parameter_std"] = perr.tolist()

    if fit_options.fixed_temperature_K is not None:
        out["Trot_fixed_K"] = float(fit_options.fixed_temperature_K)
    else:
        out["temperature_groups"] = {}
        for g, T in group_to_T_final.items():
            if g in temp_structure["fixed_group_temperatures"]:
                T_err = np.nan
                is_fixed = True
            else:
                param_idx = n_fitted_species + n_dipole + n_fwhm + temp_structure["fitted_group_names"].index(g)
                T_err = float(T * perr[param_idx]) if perr is not None else np.nan
                is_fixed = False
            out["temperature_groups"][g] = {"Trot_K": float(T), "Trot_err_K": T_err, "is_fixed": is_fixed}

    if perr is not None and dipole_targets:
        dipole_offset = n_fitted_species
        for i, target in enumerate(dipole_targets):
            scale = dipole_scale_final[target]
            key = _dipole_target_key(target)
            out["dipole_scales"][key]["scale_err"] = float(scale * perr[dipole_offset + i])

    for i, name in enumerate(species_names):
        scale = float(A_map[name])
        if name in fixed_species:
            scale_err = np.nan
            log_scale = float(np.log(scale))
            log_scale_err = np.nan
            is_fixed = True
        else:
            fit_idx = fitted_species.index(name)
            log_scale = float(res.x[fit_idx])
            log_scale_err = float(perr[fit_idx]) if perr is not None else np.nan
            scale_err = float(scale * perr[fit_idx]) if perr is not None else np.nan
            is_fixed = False

        item = {
            "log_scale": log_scale,
            "log_scale_err": log_scale_err,
            "scale": scale,
            "scale_err": scale_err,
            "scale_fixed": is_fixed,
            "relative_abundance": float(rel_sum[i]),
            "population_relmax": float(rel_max[i]),
            "Trot_K": float(Ts_final[name]),
        }
        if fit_options.fixed_temperature_K is not None:
            item["temperature_group"] = "GLOBAL_FIXED"
            item["Trot_err_K"] = np.nan
        else:
            g = temp_structure["species_to_group"][name]
            item["temperature_group"] = g
            item["Trot_err_K"] = out["temperature_groups"][g]["Trot_err_K"]
        out["species"][name] = item

    _add_scale_monte_carlo(out, res.x, cov, species_names, fitted_species, fixed_species, fit_options)
    out["warnings"] = _build_fit_warnings(out, species_tables, dipole_targets)
    return out


def _spectrum_points_for_windows(spectrum_df, centers: np.ndarray, window_MHz: float) -> tuple[np.ndarray, np.ndarray]:
    x = spectrum_df["freq_MHz"].to_numpy(dtype=float)
    y = spectrum_df["intensity"].to_numpy(dtype=float)
    mask = np.zeros_like(x, dtype=bool)
    for center in centers:
        mask |= (x >= center - window_MHz) & (x <= center + window_MHz)
    if not np.any(mask):
        raise ValueError("Spectrum fit selected no spectrum points. Increase lineshape.window_MHz.")
    return x[mask], y[mask]


def _fit_mixture_dipole_second_step(
    species_tables: dict,
    spectrum_df,
    fit_options: FitOptions,
    q_functions: dict[str, Callable[[float], float]],
    dipole_targets: list[tuple[str, str]],
) -> dict:
    first_options = replace(
        fit_options,
        dipole_model="from_cat",
        fit_dipole_components=[],
        dipole_fit_second_step=False,
        n_mc=0,
    )
    first = fit_mixture(species_tables, spectrum_df, first_options, q_functions=q_functions)

    species_names = list(species_tables.keys())
    species_index = {name: i for i, name in enumerate(species_names)}
    obs = build_observation_table(species_tables, spectrum_df, fit_options)
    obs_species_idx = obs["species"].map(species_index).to_numpy(dtype=int)

    y_exp_line = obs["exp_intensity"].to_numpy(dtype=float)
    sigma_log = max(np.log(1.0 + float(fit_options.intensity_uncertainty_fraction)), 1e-12)
    reject_sigma = float(fit_options.dipole_fit_reject_sigma)
    min_lines = int(fit_options.dipole_fit_min_lines)

    fit_mode = str(fit_options.fit_mode).lower()
    lopts = lineshape_options(fit_options)
    fwhm = float(first.get("fit_stats", {}).get("lineshape_fwhm_MHz", lopts["fwhm_MHz"]))
    spec_x = spec_y = None
    if fit_mode == "spectrum":
        spec_x, spec_y = _spectrum_points_for_windows(spectrum_df, obs["lin_freq_MHz"].to_numpy(dtype=float), lopts["window_MHz"])
        y_exp = spec_y
    else:
        y_exp = y_exp_line

    A_map = {name: float(first["species"][name]["scale"]) for name in species_names}
    Ts = {name: float(first["species"][name]["Trot_K"]) for name in species_names}
    first_y_model_line = np.asarray(first["y_model_line"], dtype=float)
    dipole_line_mask = _dipole_target_line_mask(obs, dipole_targets)
    ratio = np.full_like(y_exp_line, np.nan, dtype=float)
    valid_ratio = dipole_line_mask & np.isfinite(y_exp_line) & np.isfinite(first_y_model_line) & (first_y_model_line > 0)
    ratio[valid_ratio] = y_exp_line[valid_ratio] / first_y_model_line[valid_ratio]

    theta_lb = 2.0 * np.log(0.05)
    theta_ub = 2.0 * np.log(20.0)
    used_line_mask = np.zeros_like(dipole_line_mask, dtype=bool)
    rejected_mask = np.zeros_like(dipole_line_mask, dtype=bool)
    z_after_prelim = np.full_like(y_exp_line, np.nan, dtype=float)
    ratio_after_prelim = np.full_like(y_exp_line, np.nan, dtype=float)
    preliminary_intensity_scale = np.full_like(y_exp_line, np.nan, dtype=float)
    dipole_scale_final: dict[tuple[str, str], float] = {}
    dipole_scale_err: dict[tuple[str, str], float] = {}
    dipole_group_diagnostics: list[dict] = []
    final_residual_parts: list[np.ndarray] = []
    nfev_total = 0
    optimality = 0.0

    for target in dipole_targets:
        group_mask = _dipole_target_line_mask(obs, [target])
        valid = group_mask & np.isfinite(y_exp_line) & np.isfinite(first_y_model_line) & (y_exp_line > 0.0) & (first_y_model_line > 0.0)
        n_total = int(group_mask.sum())
        n_valid = int(valid.sum())
        status = "ok"
        log_s0 = 0.0
        log_s = 0.0
        red_chi2 = np.nan
        se_log_s = np.nan
        median_ratio_step1 = float(np.nanmedian(ratio[valid])) if n_valid else np.nan

        def log_residual(theta: np.ndarray, mask: np.ndarray) -> np.ndarray:
            return (np.log(y_exp_line[mask]) - np.log(first_y_model_line[mask]) - float(theta[0])) / sigma_log

        if n_valid == 0:
            keep = np.zeros_like(group_mask, dtype=bool)
            status = "no_positive_lines"
        else:
            res0 = least_squares(
                log_residual,
                x0=np.array([0.0]),
                args=(valid,),
                bounds=([theta_lb], [theta_ub]),
                method="trf",
                loss="soft_l1",
                f_scale=1.0,
            )
            nfev_total += int(res0.nfev)
            optimality = max(optimality, float(res0.optimality))
            log_s0 = float(res0.x[0])
            z = log_residual(np.array([log_s0]), valid)
            z_after_prelim[valid] = z
            preliminary_intensity_scale[valid] = float(np.exp(log_s0))
            ratio_after_prelim[valid] = ratio[valid] / float(np.exp(log_s0))
            keep = valid.copy()
            if fit_options.dipole_fit_reject_outliers:
                keep[valid] = np.abs(z) <= reject_sigma

            if int(keep.sum()) < min_lines:
                status = "insufficient_accepted_lines"
                keep[:] = False
            else:
                res1 = least_squares(
                    log_residual,
                    x0=np.array([log_s0]),
                    args=(keep,),
                    bounds=([theta_lb], [theta_ub]),
                    method="trf",
                    loss="linear",
                )
                nfev_total += int(res1.nfev)
                optimality = max(optimality, float(res1.optimality))
                log_s = float(res1.x[0])
                z_final = log_residual(np.array([log_s]), keep)
                final_residual_parts.append(z_final)
                dof_group = max(int(keep.sum()) - 1, 1)
                red_chi2 = float(np.sum(z_final ** 2) / dof_group)
                se_log_s = float(sigma_log / np.sqrt(int(keep.sum())) * np.sqrt(red_chi2))

        used_line_mask |= keep
        rejected_mask |= group_mask & ~keep
        mu_scale = float(np.exp(0.5 * log_s)) if status == "ok" else 1.0
        dipole_scale_final[target] = mu_scale
        dipole_scale_err[target] = float(0.5 * mu_scale * se_log_s) if np.isfinite(se_log_s) else np.nan
        dipole_group_diagnostics.append({
            "species": target[0],
            "component": target[1],
            "status": status,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_used": int(keep.sum()),
            "n_rejected": int(n_total - keep.sum()),
            "median_ratio_step1": median_ratio_step1,
            "prelim_intensity_scale": float(np.exp(log_s0)),
            "prelim_mu_scale": float(np.exp(0.5 * log_s0)),
            "final_intensity_scale": float(mu_scale ** 2),
            "final_mu_scale": mu_scale,
            "mu_scale_err": dipole_scale_err[target],
            "red_chi2": red_chi2,
        })

    I_cat_all = obs["I_cat"].to_numpy(dtype=float)
    I_cat_unit_all = obs.get("I_cat_unit_dipole", obs["I_cat"]).to_numpy(dtype=float)
    component_all = obs.get("dipole_component", "").astype(str).to_numpy()
    dipole_used_all = obs.get("dipole_used_D", np.nan).to_numpy(dtype=float)
    Elo_all = obs["elo_cm-1"].to_numpy(dtype=float)
    DR_all = obs["dr"].to_numpy(dtype=float)
    nu_all = obs["cat_freq_MHz"].to_numpy(dtype=float)
    Tcat_all = obs["cat_temperature_K"].to_numpy(dtype=float)

    def evaluate_model(dipole_scale: dict[tuple[str, str], float]):
        y_model_line = np.zeros_like(y_exp_line, dtype=float)

        for i, name in enumerate(species_names):
            mask = obs_species_idx == i
            if not np.any(mask):
                continue

            base_I_cat = I_cat_all[mask].copy()
            unit = I_cat_unit_all[mask]
            comps = component_all[mask]
            mu_used = dipole_used_all[mask]
            valid_unit = np.isfinite(unit) & np.isfinite(mu_used)
            base_I_cat[valid_unit] = unit[valid_unit] * mu_used[valid_unit] ** 2

            for target, scale in dipole_scale.items():
                target_species, component = target
                if target_species not in {"*", name}:
                    continue
                cmask = valid_unit & (comps == component)
                base_I_cat[cmask] = unit[cmask] * (mu_used[cmask] * scale) ** 2

            if fit_options.use_exact_temperature_scaling and name in q_functions:
                pred = intensity_at_T_exact(base_I_cat, Elo_all[mask], nu_all[mask], Ts[name], Tcat_all[mask], q_functions[name])
            else:
                pred = intensity_at_T_approx(base_I_cat, Elo_all[mask], DR_all[mask], Ts[name], Tcat_all[mask])
            y_model_line[mask] = A_map[name] * pred

        return y_model_line

    y_model_line = evaluate_model(dipole_scale_final)
    if fit_mode == "spectrum":
        y_model = render_profiles(
            spec_x,
            obs["lin_freq_MHz"].to_numpy(dtype=float),
            y_model_line,
            fwhm=fwhm,
            profile=lopts["profile"],
            normalize=lopts["normalize"],
        )
    else:
        y_model = y_model_line

    r = np.concatenate(final_residual_parts) if final_residual_parts else np.array([], dtype=float)
    raw_residual = y_exp - y_model
    raw_line_residual = y_exp_line - y_model_line
    raw_line_residual_used = raw_line_residual[used_line_mask]
    n_obs = int(r.size)
    n_par = int(sum(1 for item in dipole_group_diagnostics if item["status"] == "ok"))
    dof = max(n_obs - n_par, 1)
    rss = float(np.sum(r ** 2))
    sigma2 = float(rss / dof)
    unit_weight_rms = float(np.sqrt(sigma2))
    objective_rmse_nobs = float(np.sqrt(np.mean(r ** 2))) if n_obs else np.nan
    max_abs_weighted_residual = float(np.max(np.abs(r))) if n_obs else np.nan
    mean_weighted_residual = float(np.mean(r)) if n_obs else np.nan

    out = copy.deepcopy(first)
    out.update({
        "success": bool(n_par > 0),
        "status": 2 if n_par > 0 else 0,
        "message": "second-step dipole fit: per-component robust log preliminary fit and accepted-line refit completed"
        if n_par > 0 else "second-step dipole fit: no component had enough accepted positive-intensity lines",
        "cost": float(0.5 * rss),
        "nfev": int(nfev_total),
        "optimality": float(optimality),
        "x0": [0.0] * len(dipole_targets),
        "x_final": [float(np.log(dipole_scale_final[target])) for target in dipole_targets],
        "residual_vector": r.tolist(),
        "raw_residual_vector": raw_residual.tolist(),
        "y_exp": y_exp.tolist(),
        "y_model": y_model.tolist(),
        "y_exp_line": y_exp_line.tolist(),
        "y_model_line": y_model_line.tolist(),
        "observation_table": obs,
        "dipole_scales": _format_dipole_scales(dipole_scale_final),
        "dipole_group_diagnostics": dipole_group_diagnostics,
        "first_step_result": first,
    })

    out["fit_stats"] = dict(first["fit_stats"])
    out["fit_stats"].update({
        "n_obs": n_obs,
        "n_par": n_par,
        "dof": int(dof),
        "rss": rss,
        "rmse": unit_weight_rms,
        "unit_weight_rms": unit_weight_rms,
        "objective_rmse_nobs": objective_rmse_nobs,
        "raw_rmse": float(np.sqrt(np.mean(raw_residual ** 2))),
        "residual_std": float(np.std(r, ddof=1)) if n_obs > 1 else np.nan,
        "max_abs_weighted_residual": max_abs_weighted_residual,
        "mean_weighted_residual": mean_weighted_residual,
        "raw_mean_residual": float(np.mean(raw_residual)),
        "raw_max_abs_residual": float(np.max(np.abs(raw_residual))),
        "sigma2": sigma2,
        "cond_JTJ": np.nan,
        "fit_mode": fit_mode,
        "line_raw_rmse": float(np.sqrt(np.mean(raw_line_residual_used ** 2))) if raw_line_residual_used.size else np.nan,
        "line_raw_rmse_all": float(np.sqrt(np.mean(raw_line_residual ** 2))),
        "dipole_fit_second_step": True,
        "dipole_fit_residual_model": "natural_log_intensity",
        "dipole_fit_residual_weighting": "log_fractional",
        "dipole_fit_sigma_log": sigma_log,
        "dipole_fit_reject_outliers": bool(fit_options.dipole_fit_reject_outliers),
        "dipole_fit_reject_sigma": reject_sigma,
        "dipole_fit_min_lines": min_lines,
        "dipole_fit_candidate_lines": int(dipole_line_mask.sum()),
        "dipole_fit_used_lines": int(used_line_mask.sum()),
        "dipole_fit_rejected_lines": int(rejected_mask.sum()),
        "first_step_n_par": first["fit_stats"]["n_par"],
        "first_step_rmse": first["fit_stats"]["rmse"],
        "n_mc": 0,
    })

    out.pop("covariance_matrix", None)
    out.pop("parameter_std", None)

    for target in dipole_targets:
        key = _dipole_target_key(target)
        out["dipole_scales"][key]["scale_err"] = dipole_scale_err.get(target, np.nan)

    out["dipole_rejected_lines"] = _rejected_dipole_lines(
        obs,
        first_y_model_line,
        ratio,
        rejected_mask,
        z_after_prelim=z_after_prelim,
        ratio_after_prelim=ratio_after_prelim,
        preliminary_intensity_scale=preliminary_intensity_scale,
    )
    out["warnings"] = [
        "Second-step dipole fit: abundance, relative populations, temperatures, and FWHM are fixed to the first-step fitted values."
        " Dipole scales are conditional intensity corrections, not independent absolute dipole measurements."
    ] + _build_fit_warnings(out, species_tables, dipole_targets, second_step=True)
    for item in dipole_group_diagnostics:
        if item["status"] != "ok":
            label = f"{item['species']}:mu_{item['component']}"
            out["warnings"].append(f"{label}: second-step dipole scale left at 1.0 because status={item['status']}.")
        elif item["n_total"] and item["n_rejected"] / item["n_total"] > 0.4:
            label = f"{item['species']}:mu_{item['component']}"
            out["warnings"].append(f"{label}: rejected {item['n_rejected']}/{item['n_total']} lines after component correction.")
        if item["status"] == "ok" and (item["final_mu_scale"] > 1.5 or item["final_mu_scale"] < 0.67):
            label = f"{item['species']}:mu_{item['component']}"
            out["warnings"].append(
                f"{label}: large conditional mu scale {item['final_mu_scale']:.3f}; check abundance x mu^2 degeneracy, blends, response correction, and .int dipoles."
            )
    return out


def _build_dipole_targets(species_tables: dict, fit_options: FitOptions) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []

    for name, tab in species_tables.items():
        for component in tab.attrs.get("fit_dipole_components", []) or []:
            component = str(component).lower()
            if component not in {"a", "b", "c"}:
                raise ValueError(f"{name}: fit_dipole_components can contain only 'a', 'b', or 'c'")
            targets.append((name, component))

    if not targets:
        for component in fit_options.fit_dipole_components or []:
            component = str(component).lower()
            if component not in {"a", "b", "c"}:
                raise ValueError("fit.fit_dipole_components can contain only 'a', 'b', or 'c'")
            targets.append(("*", component))

    seen = set()
    unique_targets: list[tuple[str, str]] = []
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        unique_targets.append(target)
    return unique_targets


def _dipole_target_line_mask(obs, dipole_targets: list[tuple[str, str]]) -> np.ndarray:
    species = obs["species"].astype(str).to_numpy()
    components = obs["dipole_component"].astype(str).to_numpy()
    mask = np.zeros(len(obs), dtype=bool)
    for target_species, component in dipole_targets:
        if target_species == "*":
            mask |= components == component
        else:
            mask |= (species == target_species) & (components == component)
    return mask


def _rejected_dipole_lines(
    obs,
    first_model: np.ndarray,
    ratio: np.ndarray,
    rejected_mask: np.ndarray,
    z_after_prelim: np.ndarray | None = None,
    ratio_after_prelim: np.ndarray | None = None,
    preliminary_intensity_scale: np.ndarray | None = None,
) -> list[dict]:
    rows = []
    if not np.any(rejected_mask):
        return rows
    rejected = obs.loc[rejected_mask].copy()
    for idx, row in rejected.iterrows():
        z_val = np.nan if z_after_prelim is None else z_after_prelim[idx]
        ratio_prelim_val = np.nan if ratio_after_prelim is None else ratio_after_prelim[idx]
        prelim_scale_val = np.nan if preliminary_intensity_scale is None else preliminary_intensity_scale[idx]
        if not np.isfinite(row["exp_intensity"]) or float(row["exp_intensity"]) <= 0:
            reason = "nonpositive_observed_intensity"
        elif not np.isfinite(first_model[idx]) or float(first_model[idx]) <= 0:
            reason = "nonpositive_step1_model"
        elif np.isfinite(z_val):
            reason = "large_log_residual_after_component_prelim_fit"
        else:
            reason = "component_fit_not_available"
        rows.append({
            "species": str(row["species"]),
            "component": str(row.get("dipole_component", "unknown")),
            "lin_freq_MHz": float(row["lin_freq_MHz"]),
            "exp_intensity": float(row["exp_intensity"]),
            "step1_model_intensity": float(first_model[idx]),
            "ratio_exp_step1": float(ratio[idx]),
            "prelim_intensity_scale": float(prelim_scale_val) if np.isfinite(prelim_scale_val) else np.nan,
            "ratio_after_prelim": float(ratio_prelim_val) if np.isfinite(ratio_prelim_val) else np.nan,
            "z_after_prelim": float(z_val) if np.isfinite(z_val) else np.nan,
            "reject_reason": reason,
            "QN_all": row.get("QN_all"),
        })
    return rows


def _dipole_target_key(target: tuple[str, str]) -> str:
    species, component = target
    return f"{species}:mu_{component}" if species != "*" else f"GLOBAL:mu_{component}"


def _format_dipole_scales(dipole_scale: dict[tuple[str, str], float]) -> dict:
    out = {}
    for target, scale in dipole_scale.items():
        species, component = target
        out[_dipole_target_key(target)] = {
            "species": species,
            "component": component,
            "scale": scale,
            "scale_err": np.nan,
        }
    return out


def _add_scale_monte_carlo(out, x, cov, species_names, fitted_species, fixed_species, fit_options):
    if cov is None or fixed_species or fit_options.n_mc <= 0:
        out["fit_stats"]["n_mc"] = 0
        return

    n_fitted_species = len(fitted_species)
    rng = np.random.default_rng(fit_options.random_seed)
    try:
        samples = rng.multivariate_normal(mean=x, cov=cov, size=fit_options.n_mc)
        samples[:, :n_fitted_species] = np.clip(samples[:, :n_fitted_species], -50.0, 50.0)
        mc_A_fit = np.exp(samples[:, :n_fitted_species])
        mc_A = np.zeros((fit_options.n_mc, len(species_names)), dtype=float)
        for j, name in enumerate(species_names):
            mc_A[:, j] = mc_A_fit[:, fitted_species.index(name)]

        mc_rel_sum = mc_A / np.sum(mc_A, axis=1, keepdims=True)
        mc_rel_max = mc_A / np.max(mc_A, axis=1, keepdims=True)

        for i, name in enumerate(species_names):
            scale_center = out["species"][name]["scale"]
            relsum_center = out["species"][name]["relative_abundance"]
            relmax_center = out["species"][name]["population_relmax"]
            out["species"][name]["scale_mc_rms"] = float(np.sqrt(np.mean((mc_A[:, i] - scale_center) ** 2)))
            out["species"][name]["relative_abundance_mc_rms"] = float(
                np.sqrt(np.mean((mc_rel_sum[:, i] - relsum_center) ** 2))
            )
            out["species"][name]["population_relmax_mc_rms"] = float(
                np.sqrt(np.mean((mc_rel_max[:, i] - relmax_center) ** 2))
            )
        out["fit_stats"]["n_mc"] = int(fit_options.n_mc)
    except Exception:
        out["fit_stats"]["n_mc"] = 0


def _build_fit_warnings(out: dict, species_tables: dict, dipole_targets: list[tuple[str, str]], second_step: bool = False) -> list[str]:
    warnings: list[str] = []
    if dipole_targets:
        if not second_step:
            warnings.append(
                "Dipole component scales can be strongly correlated with abundance because intensities scale as abundance * mu^2."
            )
        for target_species, component in dipole_targets:
            names = species_tables if target_species == "*" else {target_species: species_tables[target_species]}
            for name, tab in names.items():
                n_present = int((tab.get("dipole_component", []) == component).sum())
                if n_present == 0:
                    lin_counts = tab.attrs.get("lin_filtered_component_counts", {})
                    missing_counts = tab.attrs.get("missing_qn_component_counts", {})
                    n_lin = int(lin_counts.get(component, 0))
                    n_missing = int(missing_counts.get(component, 0))
                    if n_lin > 0 and n_missing == n_lin:
                        warnings.append(
                            f"{name}: requested fitted mu_{component} scale; {n_lin} {component}-type LIN lines pass filters, "
                            "but none have exact QN matches in CAT, so the component cannot be fitted."
                        )
                    else:
                        warnings.append(f"{name}: requested fitted mu_{component} scale, but no matched {component}-type fitted lines are present.")
                if component not in (tab.attrs.get("dipoles", {}) or {}):
                    warnings.append(
                        f"{name}: requested fitted mu_{component} scale, but no mu_{component} value was read from the .int file."
                    )
                if not second_step and tab.attrs.get("fixed_scale", None) is None:
                    warnings.append(
                        f"{name}: mu_{component} is fitted while abundance/scale is also free; the two can be highly correlated."
                    )
    return warnings
