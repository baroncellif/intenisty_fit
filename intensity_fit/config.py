from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SpeciesConfig:
    name: str
    lin_file: str
    cat_file: str
    int_file: str | None = None
    str_file: str | None = None
    egy_file: str | None = None
    out_file: str | None = None
    var_file: str | None = None
    filters: dict[str, Any] | None = None
    fixed_scale: float | None = None
    fit_dipole_components: list[str] | None = None
    delta_E_cm1: float | None = None  # legacy alias of dE0, always in cm-1
    conformer_degeneracy: float = 1.0
    dEe: float | None = None  # electronic energy relative to the reference conformer, in conformer_population_model.unit
    dE0: float | None = None  # ZPE-corrected energy, same unit
    exclude_from_Tconf_fit: bool = False
    label: str | None = None  # name shown in reports/CSV (e.g. the name used in the article)
    group: str | None = None  # free label (e.g. monomers, complexes); selects species for the conformer analysis


@dataclass
class FitOptions:
    fit_mode: str = "line_intensity"  # line_intensity, spectrum
    common_temperature: bool = True
    extraction_mode: str = "local_max"
    extraction_window_MHz: float = 0.05
    correct_correlated_residuals: bool = True  # spectrum fit: multiply the covariance by the residual autocorrelation factor tau
    blend_tolerance_MHz: float = 0.03  # line_intensity: transitions closer than this are ONE observed peak (0 = legacy, one datum per transition)
    fit_in_log_space: bool = False
    use_exact_temperature_scaling: bool = False
    temp_init_K: float = 2.0
    temp_min_K: float = 0.3
    temp_max_K: float = 20.0
    fixed_temperature_K: float | None = None
    species_temperature_groups: dict[str, str] | None = None
    fixed_group_temperatures: dict[str, float] | None = None
    n_mc: int = 5000
    random_seed: int = 12345
    intensity_uncertainty_fraction: float = 0.20
    intensity_uncertainty_floor: float | None = None
    residual_weighting: str = "fractional"  # fractional, hybrid, none
    loss: str = "linear"
    loss_f_scale: float = 1.0
    dipole_model: str = "from_cat"  # from_cat, component_scaled
    fit_dipole_components: list[str] | None = None
    dipole_fit_second_step: bool = False
    dipole_fit_residual_weighting: str = "fractional"  # fractional, none
    dipole_fit_reject_outliers: bool = True
    dipole_fit_outlier_factor: float = 5.0
    dipole_fit_reject_sigma: float = 5.0
    dipole_fit_min_lines: int = 3
    conformer_population_model: dict[str, Any] | None = None
    lineshape: dict[str, Any] | None = None
    plot: dict[str, Any] | None = None


@dataclass
class RunConfig:
    name: str
    spectrum_path: str
    output_dir: str
    species: list[SpeciesConfig]
    fit_options: FitOptions
    plot: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


def _read_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML. Use JSON or install PyYAML.") from exc
        return dict(yaml.safe_load(text))

    if suffix == ".jsonc":
        text = _strip_jsonc_comments(text)
        text = _strip_jsonc_trailing_commas(text)

    return json.loads(text)


def _strip_jsonc_comments(text: str) -> str:
    out: list[str] = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue

        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _strip_jsonc_trailing_commas(text: str) -> str:
    out: list[str] = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]

        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] in "}]":
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def load_run_config(path: str | Path) -> RunConfig:
    raw = _read_config(path)
    run = raw.get("run", {})
    fit = raw.get("fit", {})
    species_raw = raw.get("species", [])

    if not run.get("name"):
        raise ValueError("Config is missing run.name")
    if not run.get("spectrum"):
        raise ValueError("Config is missing run.spectrum")
    if not species_raw:
        raise ValueError("Config must contain at least one species")

    species = []
    for item in species_raw:
        prefix = item.get("prefix")
        if prefix:
            lin_file = item.get("lin") or item.get("lin_file") or f"{prefix}.lin"
            cat_file = item.get("cat") or item.get("cat_file") or f"{prefix}.cat"
            int_file = item.get("int") or item.get("int_file") or f"{prefix}.int"
            str_file = item.get("str") or item.get("str_file") or f"{prefix}.str"
            egy_file = item.get("egy") or item.get("egy_file") or f"{prefix}.egy"
            out_file = item.get("out") or item.get("out_file") or f"{prefix}.out"
            var_file = item.get("var") or item.get("var_file") or f"{prefix}.var"
        else:
            lin_file = item.get("lin") or item.get("lin_file")
            cat_file = item.get("cat") or item.get("cat_file")
            int_file = item.get("int") or item.get("int_file")
            str_file = item.get("str") or item.get("str_file")
            egy_file = item.get("egy") or item.get("egy_file")
            out_file = item.get("out") or item.get("out_file")
            var_file = item.get("var") or item.get("var_file")

        species.append(
            SpeciesConfig(
                name=item["name"],
                lin_file=lin_file,
                cat_file=cat_file,
                int_file=int_file,
                str_file=str_file,
                egy_file=egy_file,
                out_file=out_file,
                var_file=var_file,
                filters=item.get("filters"),
                fixed_scale=item.get("fixed_scale"),
                fit_dipole_components=item.get("fit_dipole_components"),
                delta_E_cm1=item.get("delta_E_cm-1", item.get("delta_E_cm1")),
                conformer_degeneracy=float(item.get("conformer_degeneracy", 1.0)),
                dEe=item.get("dEe"),
                dE0=item.get("dE0"),
                exclude_from_Tconf_fit=bool(item.get("exclude_from_Tconf_fit", False)),
                group=item.get("group"),
                label=item.get("label"),
            )
        )

    for item in species:
        if not item.lin_file or not item.cat_file:
            raise ValueError(f"Species {item.name!r} must define lin and cat paths")

    fit_options = FitOptions(**fit)

    return RunConfig(
        name=run["name"],
        spectrum_path=run["spectrum"],
        output_dir=run.get("output_dir", run["name"]),
        species=species,
        fit_options=fit_options,
        plot=raw.get("plot"),
        raw=raw,
    )
