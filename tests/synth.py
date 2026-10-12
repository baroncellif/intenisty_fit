"""Synthetic Pickett files and spectra with a known ground truth.

Every species gets R-branch a-type (and optionally c-type) transitions.  The
"experimental" spectrum is built from the same intensity model the fit uses, so
a correct implementation must recover the injected scales / temperatures.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from intensity_fit.constants import C2
from intensity_fit.lineshape import render_profiles

T_CAT = 20.0
DR = 3
FWHM = 0.08
STEP = 0.01


@dataclass
class SpeciesSpec:
    name: str
    base_freq: float
    n_a: int = 5
    n_c: int = 0
    scale: float = 1.0
    T: float = 6.0
    mu_scale_c: float = 1.0
    dipoles: tuple = (1.0, 1.0, 1.0)
    with_int: bool = True
    with_out: bool = True


def transitions(spec: SpeciesSpec) -> list[dict]:
    rows = []
    for J in range(spec.n_a):
        rows.append(dict(
            qn=(J + 1, 0, J + 1, J, 0, J),
            freq=round(spec.base_freq + 100.0 * J, 2),
            lgint=-0.1 * J, elo=0.08 * J * (J + 1), comp="a",
        ))
    for k in range(spec.n_c):
        J = k + 1
        rows.append(dict(
            qn=(J + 1, 1, J, J, 0, J),
            freq=round(spec.base_freq + 100.0 * J + 40.0, 2),
            lgint=-0.1 * J - 0.05, elo=0.08 * J * (J + 1) + 0.02, comp="c",
        ))
    return rows


def cat_line(t: dict, tag: int = 1) -> str:
    head = f"{t['freq']:13.4f}{0.001:8.4f}{t['lgint']:8.4f}{DR:2d}{t['elo']:10.4f}{3:3d}{tag:7d}{303:4d}"
    return head + "".join(f"{q:2d}" for q in t["qn"])


def lin_line(t: dict, freq: float | None = None) -> str:
    qn = "".join(f"{q:3d}" for q in t["qn"]).ljust(36)
    return f"{qn}{(t['freq'] if freq is None else freq):16.4f} {0.01:8.4f} {1.0:8.4f}"


def int_text(spec: SpeciesSpec) -> str:
    a, b, c = spec.dipoles
    return (
        f"{spec.name}\n"
        f"0 1 1.0 0 40 -10.0 -1.0 100.0 {T_CAT} 25\n"
        f" 001 {a} / a dipole\n"
        f" 002 {b} / b dipole\n"
        f" 003 {c} / c dipole\n"
    )


def out_text() -> str:
    rows = "\n".join(f"{T:10.3f} {T ** 1.5:18.10f} {1.5 * np.log10(T):10.4f}" for T in (2.0, 5.0, 10.0, 20.0, 50.0, 300.0))
    return f"some header\n TEMPERATURE - Q\n{rows}\n\ntrailer\n"


def true_intensity(t: dict, spec: SpeciesSpec) -> float:
    I_cat = 10.0 ** t["lgint"]
    if t["comp"] == "c":
        I_cat *= spec.mu_scale_c ** 2
    T = spec.T
    pred = I_cat * (T_CAT / T) ** (1.0 + 0.5 * DR) * np.exp(-C2 * t["elo"] * (1.0 / T - 1.0 / T_CAT))
    return spec.scale * pred


def write_species(tmp: Path, spec: SpeciesSpec) -> str:
    rows = transitions(spec)
    prefix = tmp / spec.name
    Path(f"{prefix}.cat").write_text("\n".join(cat_line(t, i + 1) for i, t in enumerate(rows)) + "\n")
    Path(f"{prefix}.lin").write_text("\n".join(lin_line(t) for t in rows) + "\n")
    if spec.with_int:
        Path(f"{prefix}.int").write_text(int_text(spec))
    if spec.with_out:
        Path(f"{prefix}.out").write_text(out_text())
    return str(prefix)


def write_spectrum(path: Path, specs: list[SpeciesSpec], normalize: str = "peak") -> None:
    centers, amps = [], []
    for spec in specs:
        for t in transitions(spec):
            centers.append(t["freq"])
            amps.append(true_intensity(t, spec))
    offsets = np.round(np.arange(-40, 41) * STEP, 2)
    x = np.unique(np.round(np.concatenate([c + offsets for c in centers]), 2))
    y = render_profiles(x, centers, amps, fwhm=FWHM, profile="gaussian", normalize=normalize)
    np.savetxt(path, np.column_stack([x, y]), fmt="%.6f %.10e")


@dataclass
class Scenario:
    specs: list
    config: dict
    config_path: Path
    spectrum_path: Path
    out_dir: Path


def build_scenario(tmp: Path, specs: list[SpeciesSpec], fit: dict | None = None,
                   normalize: str = "peak", filters: dict | None = None) -> Scenario:
    tmp = Path(tmp)
    spec_path = tmp / "spectrum.txt"
    write_spectrum(spec_path, specs, normalize=normalize)
    species_cfg = []
    for spec in specs:
        prefix = write_species(tmp, spec)
        item = {"name": spec.name, "prefix": prefix}
        if filters is not None:
            item["filters"] = filters
        species_cfg.append(item)
    fit_cfg = {"extraction_mode": "local_max", "extraction_window_MHz": 0.05, "temp_init_K": 4.0,
               "temp_min_K": 0.5, "temp_max_K": 50.0, "n_mc": 200}
    fit_cfg.update(fit or {})
    out_dir = tmp / "out"
    config = {"run": {"name": "synthetic", "spectrum": str(spec_path), "output_dir": str(out_dir)},
              "fit": fit_cfg, "species": species_cfg, "plot": {"enabled": False}}
    config_path = tmp / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    return Scenario(specs, config, config_path, spec_path, out_dir)
