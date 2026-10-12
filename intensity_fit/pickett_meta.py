from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def existing_file(path: str | None) -> str | None:
    if not path:
        return None
    return path if Path(path).exists() else None


def read_int_metadata(filename: str | None) -> dict:
    path = existing_file(filename)
    if path is None:
        return {"exists": False, "dipoles": {}, "raw": []}

    lines = Path(path).read_text(errors="replace").splitlines()
    out = {
        "exists": True,
        "title": lines[0].strip() if lines else "",
        "raw": lines,
        "dipoles": {},
    }

    if len(lines) >= 2:
        vals = _numbers(lines[1])
        keys = ["iflg", "tag", "qrot_input", "qn_min", "qn_max", "str0", "str1", "freq_max_GHz", "temperature_K", "maxv"]
        for key, val in zip(keys, vals):
            out[key] = val

    for line in lines[2:]:
        vals = _numbers(line)
        if len(vals) < 2:
            continue
        dip_id = int(vals[0])
        value = float(vals[1])
        comment = line.split("/", 1)[1].strip().lower() if "/" in line else ""
        component = None
        if "a dipole" in comment or comment.startswith("a"):
            component = "a"
        elif "b dipole" in comment or comment.startswith("b"):
            component = "b"
        elif "c dipole" in comment or comment.startswith("c"):
            component = "c"
        elif dip_id == 1:
            component = "a"
        elif dip_id == 2:
            component = "b"
        elif dip_id == 3:
            component = "c"
        if component:
            out["dipoles"][component] = value

    return out


def read_out_partition_function(filename: str | None) -> pd.DataFrame:
    path = existing_file(filename)
    if path is None:
        return pd.DataFrame(columns=["T_K", "Q", "logQ"])

    rows = []
    in_table = False
    for line in Path(path).read_text(errors="replace").splitlines():
        if "TEMPERATURE - Q" in line:
            in_table = True
            continue
        if not in_table:
            continue
        vals = _numbers(line)
        if len(vals) >= 3:
            rows.append({"T_K": float(vals[0]), "Q": float(vals[1]), "logQ": float(vals[2])})
        elif rows:
            break
    return pd.DataFrame(rows)


def make_partition_function(q_table: pd.DataFrame):
    if q_table.empty:
        return None
    q_table = q_table.sort_values("T_K")
    T = q_table["T_K"].to_numpy(dtype=float)
    Q = q_table["Q"].to_numpy(dtype=float)
    mask = (T > 0) & (Q > 0)
    T = T[mask]
    Q = Q[mask]
    if T.size < 2:
        return None

    logT = np.log(T)
    logQ = np.log(Q)

    def qfunc(temp: float) -> float:
        temp = float(temp)
        if temp <= 0:
            return np.nan
        return float(np.exp(np.interp(np.log(temp), logT, logQ)))

    return qfunc


def read_egy(filename: str | None) -> pd.DataFrame:
    path = existing_file(filename)
    if path is None:
        return pd.DataFrame()
    # EGY output format depends strongly on SPCAT flags. Keep a conservative
    # numeric-row parser for diagnostics; detailed semantic parsing can be
    # extended once a representative .egy file is available.
    rows = []
    for line_no, line in enumerate(Path(path).read_text(errors="replace").splitlines(), start=1):
        vals = _numbers(line)
        if len(vals) >= 2:
            rows.append({"line_number": line_no, "raw": line, "values": vals})
    return pd.DataFrame(rows)


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", text)]
