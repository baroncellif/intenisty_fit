"""Comparison of fitted abundances with a frozen Boltzmann conformer distribution.

The intensity fit gives one independent abundance A_i per species.  Here those
abundances are compared with p_i ~ g_i exp(-c2 E_i / Tconf), where Tconf is the
conformational temperature the populations were frozen at (before the final
cooling in the expansion).  Tconf can be given (fixed) or fitted a posteriori,
separately for the electronic (Ee) and ZPE-corrected (E0) energies:

    ln A_i - ln g_i = c - c2 E_i / Tconf          (c = free overall normalisation)

This is linear in 1/Tconf and is solved by generalised least squares using the
covariance of ln A_i from the intensity fit, so the result does not depend on
the energy zero.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares

from .constants import C2

_KJ_MOL_CM1 = 83.5934722
UNIT_TO_CM1 = {
    "cm-1": 1.0,
    "kj/mol": _KJ_MOL_CM1,
    "kcal/mol": 4.184 * _KJ_MOL_CM1,
    "hartree": 219474.6313632,
    "ev": 8065.543937,
    "k": 1.0 / C2,
}
_UNIT_ALIASES = {"cm^-1": "cm-1", "cm1": "cm-1", "1/cm": "cm-1", "kj": "kj/mol", "kcal": "kcal/mol",
                 "eh": "hartree", "au": "hartree"}


def energy_unit_to_cm1(unit: str) -> float:
    key = str(unit).strip().lower()
    key = _UNIT_ALIASES.get(key, key)
    if key not in UNIT_TO_CM1:
        raise ValueError(f"Unknown energy unit {unit!r}; use one of: cm-1, kJ/mol, kcal/mol, hartree, eV, K")
    return UNIT_TO_CM1[key]


def _references(model: dict) -> list[str]:
    ref = str(model.get("reference", "both")).strip().lower()
    if ref == "both":
        return ["Ee", "E0"]
    if ref == "ee":
        return ["Ee"]
    if ref == "e0":
        return ["E0"]
    raise ValueError("conformer_population_model.reference must be 'both', 'Ee', or 'E0'")


def _tconf_setting(model: dict) -> tuple[str, float | None]:
    value = model.get("Tconf")
    if value is None:  # legacy keys
        if str(model.get("temperature_source", "")).lower() == "fixed" and "temperature_K" in model:
            return "fixed", float(model["temperature_K"])
        return "mean_rotational", None
    if isinstance(value, str):
        key = value.strip().lower()
        if key == "fit":
            return "fit", None
        if key in {"mean_rotational", "mean_rotational_temperature"}:
            return "mean_rotational", None
        raise ValueError("conformer_population_model.Tconf must be a number, 'fit', or 'mean_rotational'")
    value = float(value)
    if value <= 0:
        raise ValueError("conformer_population_model.Tconf must be > 0")
    return "fixed", value


def _energy_cm1(sp, ref: str, factor: float) -> float | None:
    value = sp.dEe if ref == "Ee" else sp.dE0
    if value is not None:
        return float(value) * factor
    if ref == "E0" and sp.delta_E_cm1 is not None:  # legacy key, always cm-1
        return float(sp.delta_E_cm1)
    return None


def _log_scale_covariance(result: dict, names: list[str]) -> np.ndarray | None:
    """Covariance of ln(scale) for the given species, or None if unavailable."""
    fitted = [n for n, v in result["species"].items() if not v.get("scale_fixed", False)]
    cov_full = result.get("covariance_matrix")
    idx = [fitted.index(n) for n in names]
    if cov_full is not None:
        cov = np.asarray(cov_full, dtype=float)[np.ix_(idx, idx)]
        if np.all(np.isfinite(cov)):
            return cov
    errs = np.array([result["species"][n].get("log_scale_err", np.nan) for n in names], dtype=float)
    if np.all(np.isfinite(errs)):
        return np.diag(errs ** 2)
    return None


def _whitener(cov: np.ndarray | None, n: int) -> np.ndarray:
    if cov is None:
        return np.eye(n)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        diag = np.diag(cov)
        if np.all(diag > 0):
            return np.diag(1.0 / np.sqrt(diag))
        return np.eye(n)
    return solve_triangular(chol, np.eye(n), lower=True)


def _fit_tconf(x, y, whiten, T_start, T_min, T_max) -> dict:
    """Generalised least squares for  y = c - x*beta,  beta = 1/Tconf  (bounded, convex)."""
    n = len(y)
    beta0 = float(np.clip(1.0 / T_start, 1.0 / T_max, 1.0 / T_min))
    c0 = float(np.mean(y + x * beta0))

    def residuals(p):
        return whiten @ (y - (p[0] - x * p[1]))

    res = least_squares(residuals, [c0, beta0], bounds=([-np.inf, 1.0 / T_max], [np.inf, 1.0 / T_min]))
    c, beta = float(res.x[0]), float(res.x[1])
    rss = float(np.sum(res.fun ** 2))
    dof = n - 2
    T = 1.0 / beta
    T_err = np.nan
    if dof > 0:
        cov_p = (rss / dof) * np.linalg.pinv(res.jac.T @ res.jac)
        T_err = float(np.sqrt(cov_p[1, 1]) * T ** 2)
    at_bound = bool(np.isclose(beta, 1.0 / T_max, rtol=1e-6) or np.isclose(beta, 1.0 / T_min, rtol=1e-6))
    return {"Tconf_K": T, "Tconf_err_K": T_err, "c": c, "rss": rss, "dof": dof,
            "chi2_red": rss / dof if dof > 0 else np.nan, "at_bound": at_bound}


def _analyze_set(species_cfgs: list, model: dict, result: dict, fixed_T_by_ref: dict | None = None,
                 source_label: str | None = None, group: str | None = None) -> dict:
    """One Tconf analysis of a set of species (all species, or one group), for every requested energy reference."""
    factor = energy_unit_to_cm1(model.get("unit", "cm-1"))
    refs = _references(model)
    tsource, tfixed = ("fixed", None) if fixed_T_by_ref is not None else _tconf_setting(model)
    T_start = float(model.get("Tconf_start", 300.0))
    T_min = float(model.get("Tconf_min", 10.0))
    T_max = float(model.get("Tconf_max", 1000.0))
    allowed = model.get("species")
    groups = model.get("groups")
    if isinstance(groups, str):
        groups = [groups]

    out = {"rows": [], "fits": {}, "warnings": [], "excluded": [], "unit": str(model.get("unit", "cm-1"))}
    out["excluded"] = [sp.name for sp in species_cfgs if sp.exclude_from_Tconf_fit]

    for ref in refs:
        ref_source, ref_tfixed = tsource, tfixed
        if fixed_T_by_ref is not None:
            if ref not in fixed_T_by_ref:
                out["warnings"].append(f"{ref}: no Tconf available from {source_label}; skipped.")
                continue
            ref_source, ref_tfixed = "fixed", fixed_T_by_ref[ref]
        members = []
        for sp in species_cfgs:
            if sp.exclude_from_Tconf_fit or sp.name not in result["species"]:
                continue
            if allowed is not None and sp.name not in allowed:
                continue
            if groups is not None and sp.group not in groups:
                continue
            energy = _energy_cm1(sp, ref, factor)
            if energy is None:
                continue
            if result["species"][sp.name].get("scale_fixed", False):
                out["warnings"].append(f"{sp.name}: fixed_scale species skipped in the {ref} conformer comparison (no uncertainty).")
                continue
            members.append((sp, energy))

        if not members:
            out["warnings"].append(f"No species with {ref} energy available for the conformer comparison.")
            continue

        names = [sp.name for sp, _ in members]
        E = np.array([e for _, e in members], dtype=float)
        g = np.array([sp.conformer_degeneracy for sp, _ in members], dtype=float)
        A = np.array([result["species"][n]["scale"] for n in names], dtype=float)
        cov = _log_scale_covariance(result, names)
        if cov is None:
            out["warnings"].append(f"{ref}: no abundance uncertainties available; conformer fit is unweighted.")
        x = C2 * E

        fit_info = {"source": source_label or ref_source, "n_species": len(names)}
        if ref_source == "fit":
            if len(names) < 2:
                out["warnings"].append(f"{ref}: Tconf fit needs at least 2 species, got {len(names)}.")
                continue
            fit_info.update(_fit_tconf(x, np.log(A) - np.log(g), _whitener(cov, len(names)), T_start, T_min, T_max))
            if fit_info["at_bound"]:
                out["warnings"].append(f"{ref}: fitted Tconf={fit_info['Tconf_K']:.1f} K sits on a bound [{T_min}, {T_max}] K.")
            if fit_info["dof"] <= 0:
                out["warnings"].append(f"{ref}: {len(names)} species leave no degrees of freedom; Tconf has no uncertainty.")
            Tconf = fit_info["Tconf_K"]
        elif ref_source == "fixed":
            Tconf = ref_tfixed
            fit_info.update({"Tconf_K": Tconf, "Tconf_err_K": np.nan})
        else:
            Tconf = float(np.mean([result["species"][n]["Trot_K"] for n in names]))
            fit_info.update({"Tconf_K": Tconf, "Tconf_err_K": np.nan})
        out["fits"][ref] = fit_info

        w = g * np.exp(-x / Tconf)
        expected = w / w.sum()
        fitted = A / A.sum()
        frac_err = np.full(len(names), np.nan)
        if cov is not None:
            for i in range(len(names)):
                grad = -fitted.copy()
                grad[i] += 1.0
                frac_err[i] = fitted[i] * np.sqrt(max(float(grad @ cov @ grad), 0.0))

        for i, name in enumerate(names):
            out["rows"].append({
                "group": group if group is not None else "",
                "reference": ref,
                "species": name,
                "label": members[i][0].label or name,
                "energy_cm-1": float(E[i]),
                "degeneracy": float(g[i]),
                "Tconf_K": float(Tconf),
                "Tconf_source": source_label or ref_source,
                "expected_fraction": float(expected[i]),
                "fitted_fraction": float(fitted[i]),
                "fitted_fraction_err": float(frac_err[i]),
                "fitted_over_expected": float(fitted[i] / expected[i]) if expected[i] > 0 else np.nan,
            })
    return out


def analyze_conformers(species_cfgs: list, fit_options, result: dict) -> dict:
    """Compare the fitted abundances with a frozen Boltzmann conformer distribution.

    ``per_group = false`` (default): one analysis over all species (optionally restricted with ``groups`` / ``species``).
    ``per_group = true``: the analysis is repeated for every group (``groups`` selects which; default all groups present),
    each with its own normalisation and its own Tconf.  ``group_options`` overrides options per group; a group with
    ``Tconf = "from:<other group>"`` is not fitted: its expected fractions use the Tconf fitted for that group.
    """
    model = dict(fit_options.conformer_population_model or {})
    empty = {"rows": [], "fits": {}, "warnings": [], "excluded": [], "groups": {}}
    if not model.get("enabled", False):
        return empty
    if not model.get("per_group", False):
        out = _analyze_set(species_cfgs, model, result)
        out["groups"] = {}
        return out

    present = []
    for sp in species_cfgs:
        if sp.group and sp.group not in present:
            present.append(sp.group)
    wanted = model.get("groups")
    wanted = present if wanted is None else ([wanted] if isinstance(wanted, str) else list(wanted))
    base = {k: v for k, v in model.items() if k not in ("groups", "group_options", "per_group", "species", "also_with_Tconf_from")}
    overrides = model.get("group_options") or {}
    out = {"rows": [], "fits": {}, "warnings": [], "excluded": [], "unit": str(model.get("unit", "cm-1")), "groups": {}}

    def _merge(g, res):
        out["groups"][g] = {"fits": res["fits"], "n_rows": len(res["rows"]), "unit": res.get("unit")}
        out["rows"].extend(res["rows"])
        out["warnings"].extend(f"[{g}] {w}" for w in res["warnings"])
        out["excluded"].extend(x for x in res["excluded"] if x not in out["excluded"])

    pending = []
    for g in wanted:
        opts = {**base, **{k: v for k, v in (overrides.get(g) or {}).items() if k != "also_with_Tconf_from"}}
        members = [sp for sp in species_cfgs if sp.group == g]
        if not members:
            out["warnings"].append(f"[{g}] no species in this group.")
            continue
        setting = opts.get("Tconf")
        if isinstance(setting, str) and setting.strip().lower().startswith("from:"):
            pending.append((g, opts, members, setting.split(":", 1)[1].strip()))
            continue
        res = _analyze_set(members, opts, result, group=g)
        if not res["rows"] and not res["fits"]:
            out["warnings"].append(f"[{g}] no energies available: group skipped.")
            continue
        _merge(g, res)
    for g, opts, members, src in pending:
        fits = out["groups"].get(src, {}).get("fits")
        if not fits:
            out["warnings"].append(f"[{g}] Tconf requested from group {src!r}, but that group has no Tconf; skipped.")
            continue
        fixed = {ref: info["Tconf_K"] for ref, info in fits.items()}
        _merge(g, _analyze_set(members, opts, result, fixed_T_by_ref=fixed, source_label=f"from {src}", group=g))
    for g in wanted:
        src = (overrides.get(g) or {}).get("also_with_Tconf_from")
        if not src or g not in out["groups"]:
            continue
        fits = out["groups"].get(src, {}).get("fits")
        if not fits:
            out["warnings"].append(f"[{g}] Tconf of group {src!r} requested for comparison, but that group has no Tconf; skipped.")
            continue
        opts = {**base, **(overrides.get(g) or {})}
        members = [sp for sp in species_cfgs if sp.group == g]
        fixed = {ref: info["Tconf_K"] for ref, info in fits.items()}
        res = _analyze_set(members, opts, result, fixed_T_by_ref=fixed, source_label=f"from {src}", group=g)
        out["groups"][f"{g} (Tconf from {src})"] = {"fits": res["fits"], "n_rows": len(res["rows"]), "unit": res.get("unit")}
        out["rows"].extend(res["rows"])
        out["warnings"].extend(f"[{g}] {w}" for w in res["warnings"])
    first = next(iter(out["groups"].values()), None)
    out["fits"] = dict(first["fits"]) if first else {}
    return out


def _full_log_covariance(result: dict):
    """(names, covariance of ln(scale) for every species; zero for fixed scales) or (names, None)."""
    names = list(result["species"])
    fitted = [n for n in names if not result["species"][n].get("scale_fixed", False)]
    cov_full = result.get("covariance_matrix")
    if cov_full is not None:
        sub = np.asarray(cov_full, dtype=float)[: len(fitted), : len(fitted)]
        if np.all(np.isfinite(sub)):
            C = np.zeros((len(names), len(names)))
            pos = [names.index(n) for n in fitted]
            C[np.ix_(pos, pos)] = sub
            return names, C
    errs = np.array([result["species"][n].get("log_scale_err", np.nan) for n in names], dtype=float)
    if np.all(np.isfinite(errs)):
        return names, np.diag(errs ** 2)
    return names, None


def group_fractions(species_cfgs: list, result: dict) -> dict:
    """{species: (fraction, error)}: abundance of every species relative to the sum over its own group.

    The error is the first-order propagation of the covariance of ln(scale); species without a group are omitted.
    """
    names, C = _full_log_covariance(result)
    A = np.array([result["species"][n]["scale"] for n in names], dtype=float)
    out = {}
    groups = {}
    for sp in species_cfgs:
        if sp.group and sp.name in names:
            groups.setdefault(sp.group, []).append(names.index(sp.name))
    for members in groups.values():
        total = A[members].sum()
        for i in members:
            f = A[i] / total
            err = np.nan
            if C is not None:
                g = np.zeros(len(names))
                g[members] = -A[members] / total
                g[i] += 1.0
                err = float(f * np.sqrt(max(float(g @ C @ g), 0.0)))
            out[names[i]] = (float(f), err)
    return out
