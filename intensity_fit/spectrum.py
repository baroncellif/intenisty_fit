from __future__ import annotations

import numpy as np
import pandas as pd


def read_spectrum(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, sep=r"\s+", comment="#", header=None)
    if df.shape[1] < 2:
        raise ValueError("Spectrum file must have at least 2 columns: freq intensity")

    out = pd.DataFrame({
        "freq_MHz": df.iloc[:, 0].astype(float),
        "intensity": df.iloc[:, 1].astype(float),
    })
    out["sigma"] = df.iloc[:, 2].astype(float) if df.shape[1] >= 3 else np.nan
    return out.sort_values("freq_MHz").reset_index(drop=True)


def integrate_area(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def extract_line_intensity(
    spec_df: pd.DataFrame,
    centers_MHz: np.ndarray,
    mode: str = "local_max",
    window_MHz: float = 0.05,
) -> np.ndarray:
    x = spec_df["freq_MHz"].to_numpy(dtype=float)
    y = spec_df["intensity"].to_numpy(dtype=float)
    out: list[float] = []

    for center in centers_MHz:
        mask = (x >= center - window_MHz) & (x <= center + window_MHz)
        if not np.any(mask):
            out.append(np.nan)
            continue

        xx = x[mask]
        yy = y[mask]
        if mode == "sample":
            out.append(float(yy[np.argmin(np.abs(xx - center))]))
        elif mode == "local_max":
            out.append(float(np.max(yy)))
        elif mode == "area":
            out.append(integrate_area(yy, xx))
        else:
            raise ValueError(f"Unknown extraction mode: {mode}")

    return np.asarray(out, dtype=float)
