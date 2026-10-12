from __future__ import annotations

import numpy as np
import pandas as pd


def decode_qn_2char(q: str) -> int:
    q = q.strip()
    if q == "":
        return 0
    if q.lstrip("-").isdigit():
        return int(q)
    if len(q) < 2:
        raise ValueError(f"Cannot decode QN field: {q!r}")

    letter = q[0]
    digit = int(q[1])
    if letter.islower():
        return -(ord(letter) - ord("a") + 1) * 10 - digit
    return (ord(letter) - ord("A") + 10) * 10 + digit


def parse_qn_block(qn_block: str, n_total: int | None = None) -> list[int]:
    raw = qn_block.rstrip("\n")
    chunks = [raw.ljust(24)[i:i + 2] for i in range(0, 24, 2)]

    vals: list[int] = []
    for ch in chunks:
        if ch.strip() != "":
            vals.append(decode_qn_2char(ch))

    if n_total is not None and len(vals) >= n_total:
        return vals[:n_total]

    vals_token: list[int] = []
    for tok in raw.split():
        if tok.lstrip("-").isdigit():
            vals_token.append(int(tok))

    if n_total is not None and len(vals_token) >= n_total:
        return vals_token[:n_total]

    return vals


def effective_nqn_from_qnfmt(qnfmt: int) -> int:
    nqn = int(qnfmt) % 10
    return 10 if nqn == 0 else nqn


def pad_qn_to_12(qn: list[int] | tuple[int, ...]) -> list[int]:
    vals = list(qn)
    return vals[:12] if len(vals) >= 12 else vals + [0] * (12 - len(vals))


def qn_key_from_list(qn_list: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(x) for x in qn_list)


def _qn_columns(qn_vals: list[int]) -> dict:
    nqn = len(qn_vals) // 2
    qn_up = qn_vals[:nqn]
    qn_lo = qn_vals[nqn:]
    qn_up_padded = pad_qn_to_12(qn_up)[:6]
    qn_lo_padded = pad_qn_to_12(qn_lo)[:6]

    return {
        "QN_all": qn_vals,
        "QN_up": qn_up,
        "QN_lo": qn_lo,
        "nqn": nqn,
        "QN_all_padded": pad_qn_to_12(qn_vals),
        "QN_up_padded": qn_up_padded,
        "QN_lo_padded": qn_lo_padded,
        "Jp": qn_up_padded[0],
        "Kap": qn_up_padded[1],
        "Kcp": qn_up_padded[2],
        "Vp": qn_up_padded[3],
        "F1p": qn_up_padded[4],
        "F2p": qn_up_padded[5],
        "J": qn_lo_padded[0],
        "Ka": qn_lo_padded[1],
        "Kc": qn_lo_padded[2],
        "V": qn_lo_padded[3],
        "F1": qn_lo_padded[4],
        "F2": qn_lo_padded[5],
    }


def read_cat(filename: str) -> pd.DataFrame:
    records = []

    with open(filename, "r") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                freq = float(line[0:13])
                err = float(line[13:21])
                lgint = float(line[21:29])
                dr = int(line[29:31])
                elo = float(line[31:41])
                gup = int(line[41:44])
                tag = int(line[44:51])
                qnfmt = int(line[51:55])
                nqn = effective_nqn_from_qnfmt(qnfmt)
                n_total = 2 * nqn
                qn_block = line[55:].rstrip("\n")
                qn_all = parse_qn_block(qn_block, n_total=n_total)

                if len(qn_all) < n_total:
                    raise ValueError(
                        f"line {line_number}: expected {n_total} QNs, got {len(qn_all)}"
                    )

                records.append({
                    "freq_MHz": freq,
                    "err_MHz": err,
                    "lgint": lgint,
                    "dr": dr,
                    "elo_cm-1": elo,
                    "gup": gup,
                    "tag": tag,
                    "qnfmt": qnfmt,
                    "qn_raw": qn_block,
                    **_qn_columns(qn_all[:n_total]),
                })
            except ValueError:
                continue

    return pd.DataFrame(records)


def read_lin(filename: str) -> pd.DataFrame:
    rows = []

    with open(filename, "r") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                parsed = parse_lin_line(line)
                if parsed is None:
                    continue
                qn_vals, freq, err, wt = parsed

                if not qn_vals or len(qn_vals) % 2 != 0:
                    continue
                if abs(freq) > 90000:
                    continue

                rows.append({
                    **_qn_columns(qn_vals),
                    "lin_freq_MHz": abs(freq),
                    "lin_err_MHz": abs(err),
                    "wt": wt,
                    "lin_line_number": line_number,
                })
            except Exception:
                continue

    return pd.DataFrame(rows)


def parse_lin_line(line: str) -> tuple[list[int], float, float, float] | None:
    """
    Parse one SPFIT .lin line.

    SPFIT reads a fixed-width quantum-number block first: 12 integer fields,
    each three columns wide, followed by free-form frequency/error/weight.
    This function follows that layout first and falls back to token parsing for
    hand-edited files with explicit whitespace-separated quantum numbers.
    """
    fixed = _parse_lin_fixed_12i3(line)
    if fixed is not None:
        return fixed
    return _parse_lin_tokenized(line)


def _parse_lin_fixed_12i3(line: str) -> tuple[list[int], float, float, float] | None:
    if len(line) < 37:
        return None

    qn_block = line[:36]
    rest = line[36:].split()
    if len(rest) < 2:
        return None

    qn_vals: list[int] = []
    for i in range(0, 36, 3):
        chunk = qn_block[i:i + 3]
        if chunk.strip() == "":
            continue
        try:
            qn_vals.append(int(chunk))
        except ValueError:
            return None

    if not qn_vals or len(qn_vals) % 2 != 0:
        return None

    try:
        freq = float(rest[0])
        err = float(rest[1])
        wt = float(rest[2]) if len(rest) >= 3 else 1.0
    except ValueError:
        return None

    return qn_vals, freq, err, wt


def _parse_lin_tokenized(line: str) -> tuple[list[int], float, float, float] | None:
    parts = line.split()
    if len(parts) < 3:
        return None
    try:
        freq = float(parts[-3])
        err = float(parts[-2])
        wt = float(parts[-1])
    except ValueError:
        return None

    qn_vals: list[int] = []
    for tok in parts[:-3]:
        try:
            qn_vals.append(int(tok))
        except ValueError:
            pass

    if not qn_vals or len(qn_vals) % 2 != 0:
        return None
    return qn_vals, freq, err, wt


def add_qn_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["qn_key"] = out["QN_all"].apply(qn_key_from_list)
    return out


def infer_branch(df: pd.DataFrame, branch: str) -> pd.Series:
    branch = str(branch).strip().lower()

    if branch in {"r", "q", "p"}:
        dJ = df["Jp"].astype(int) - df["J"].astype(int)
        if branch == "r":
            return dJ == 1
        if branch == "q":
            return dJ == 0
        return dJ == -1

    if branch in {"mu_a", "mu_b", "mu_c"}:
        dKa = (df["Kap"].astype(int) - df["Ka"].astype(int)).abs()
        dKc = (df["Kcp"].astype(int) - df["Kc"].astype(int)).abs()
        dKa_even = (dKa % 2) == 0
        dKc_even = (dKc % 2) == 0

        if branch == "mu_a":
            return dKa_even & ~dKc_even
        if branch == "mu_b":
            return ~dKa_even & ~dKc_even
        return ~dKa_even & dKc_even

    if branch in {"all", "*"}:
        return pd.Series(True, index=df.index)

    raise ValueError(f"Unknown branch filter: {branch}")


def classify_dipole_component(df: pd.DataFrame) -> pd.Series:
    """Classify asymmetric-top transitions as a-, b-, or c-type by QNs."""
    if df.empty:
        return pd.Series(dtype=object, index=df.index)
    dKa = (df["Kap"].astype(int) - df["Ka"].astype(int)).abs()
    dKc = (df["Kcp"].astype(int) - df["Kc"].astype(int)).abs()
    dKa_even = (dKa % 2) == 0
    dKc_even = (dKc % 2) == 0

    comp = pd.Series("unknown", index=df.index, dtype=object)
    comp[dKa_even & ~dKc_even] = "a"
    comp[~dKa_even & ~dKc_even] = "b"
    comp[~dKa_even & dKc_even] = "c"
    return comp


def apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    out = df.copy()
    if out.empty or filters is None:
        return out.reset_index(drop=True)

    if "freq_min" in filters and "freq_MHz" in out.columns:
        out = out[out["freq_MHz"] >= filters["freq_min"]]
    if "freq_max" in filters and "freq_MHz" in out.columns:
        out = out[out["freq_MHz"] <= filters["freq_max"]]
    if "elo_min" in filters and "elo_cm-1" in out.columns:
        out = out[out["elo_cm-1"] >= filters["elo_min"]]
    if "elo_max" in filters and "elo_cm-1" in out.columns:
        out = out[out["elo_cm-1"] <= filters["elo_max"]]
    if "branch_J" in filters:
        out = out[infer_branch(out, filters["branch_J"])]
    if "branch_mu" in filters:
        out = out[infer_branch(out, filters["branch_mu"])]
    if "branch" in filters:
        out = out[infer_branch(out, filters["branch"])]

    return out.reset_index(drop=True)


def check_duplicate_qn(df: pd.DataFrame, label: str, freq_col: str) -> None:
    if df.empty:
        return

    counts = df["qn_key"].value_counts()
    bad_keys = counts[counts > 1].index.tolist()
    if not bad_keys:
        return

    dup_df = df[df["qn_key"].isin(bad_keys)].copy()
    dup_df["n_same_qn"] = dup_df["qn_key"].map(counts)
    cols = [c for c in [freq_col, "QN_all", "qn_key", "n_same_qn"] if c in dup_df.columns]
    msg = dup_df.sort_values(["qn_key", freq_col])[cols].head(50).to_string(index=False)
    raise ValueError(f"{label}: duplicated qn_key values were found\n{msg}")


def match_lin_to_cat_exact(lin_df: pd.DataFrame, cat_df: pd.DataFrame) -> pd.DataFrame:
    lin = add_qn_key(lin_df)
    cat = add_qn_key(cat_df)
    check_duplicate_qn(lin, "LIN", "lin_freq_MHz")

    cat = cat.sort_values("freq_MHz").drop_duplicates(subset="qn_key", keep="first")
    keep_cols = [
        "qn_key", "freq_MHz", "err_MHz", "lgint", "dr", "elo_cm-1", "gup",
        "tag", "qnfmt", "nqn", "qn_raw", "QN_all", "QN_up", "QN_lo",
        "Jp", "Kap", "Kcp", "Vp", "F1p", "F2p", "J", "Ka", "Kc", "V", "F1", "F2",
    ]
    cat_sub = cat[[c for c in keep_cols if c in cat.columns]].copy()

    merged = lin.merge(
        cat_sub,
        on="qn_key",
        how="left",
        suffixes=("_lin", "_cat"),
        validate="one_to_one",
    ).rename(columns={"freq_MHz": "cat_freq_MHz", "err_MHz": "cat_err_MHz"})

    merged["match_mode"] = np.where(merged["cat_freq_MHz"].notna(), "QN", "MISSING_QN")
    return merged
