from __future__ import annotations

from typing import Callable

import numpy as np

from .constants import C2, CM1_PER_MHZ


def intensity_from_lgint(lgint) -> np.ndarray:
    return 10.0 ** np.asarray(lgint, dtype=float)


def intensity_at_T_approx(I_cat, Elo_cm, DR, T: float, T_cat) -> np.ndarray:
    """Move catalogue intensities computed at T_cat (the .int TEMP) to T.

    Only the lower-state population changes with temperature; with
    Q ~ T^(DR/2) and stimulated emission ~ 1/T the ratio is
    (T_cat/T)^(1+DR/2) exp(-c2 Elo (1/T - 1/T_cat)).
    """
    I_cat = np.asarray(I_cat, dtype=float)
    Elo_cm = np.asarray(Elo_cm, dtype=float)
    DR = np.asarray(DR, dtype=float)
    T_cat = np.asarray(T_cat, dtype=float)
    exponent = 1.0 + 0.5 * DR
    return I_cat * (T_cat / T) ** exponent * np.exp(-C2 * Elo_cm * (1.0 / T - 1.0 / T_cat))


def intensity_at_T_exact(
    I_cat,
    Elo_cm,
    freq_MHz,
    T: float,
    T_cat,
    Qfunc: Callable[[float], float],
) -> np.ndarray:
    """Exact SPCAT population ratio between T_cat (the .int TEMP) and T."""
    I_cat = np.asarray(I_cat, dtype=float)
    Elo_cm = np.asarray(Elo_cm, dtype=float)
    freq_MHz = np.asarray(freq_MHz, dtype=float)
    T_cat = np.asarray(T_cat, dtype=float)
    Eup_cm = Elo_cm + freq_MHz * CM1_PER_MHZ
    num = np.exp(-C2 * Elo_cm / T) - np.exp(-C2 * Eup_cm / T)
    den = np.exp(-C2 * Elo_cm / T_cat) - np.exp(-C2 * Eup_cm / T_cat)
    Q_cat = np.vectorize(Qfunc, otypes=[float])(T_cat)
    return I_cat * (Q_cat / Qfunc(T)) * (num / den)
