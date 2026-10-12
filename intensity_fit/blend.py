"""Blended line-intensity extraction for ``fit_mode = "line_intensity"``.

Transitions that sit on (nearly) the same frequency are one observed peak: the spectrum cannot tell them
apart.  Comparing each of them with the whole peak counts the peak several times and biases the fit.
Here the model of one observed peak is built the way the data were built: the broadened profiles of all the
transitions around it are summed on the spectrum's own frequency grid and the SAME extraction (local_max,
sample or area, same window) is applied to that model spectrum.  One data point per peak.
"""
from __future__ import annotations

import numpy as np

from .lineshape import lineshape_values
from .spectrum import extract_line_intensity, integrate_area


def cluster_frequencies(freqs: np.ndarray, tol_MHz: float) -> np.ndarray:
    """Single-link clustering of 1-D frequencies: neighbours closer than ``tol_MHz`` share a cluster id."""
    freqs = np.asarray(freqs, dtype=float)
    order = np.argsort(freqs, kind="stable")
    new = np.r_[0, np.diff(freqs[order]) > tol_MHz].astype(int)
    ids = np.empty(freqs.size, dtype=int)
    ids[order] = np.cumsum(new)
    return ids


class BlendOperator:
    def __init__(self, freqs, spectrum_df, mode: str, window_MHz: float, fwhm_MHz: float, profile: str, tol_MHz: float):
        self.freqs = np.asarray(freqs, dtype=float)
        self.mode = str(mode)
        self.cluster = cluster_frequencies(self.freqs, tol_MHz)
        self.n_clusters = int(self.cluster.max()) + 1 if self.freqs.size else 0
        self.n_in_cluster = np.bincount(self.cluster, minlength=self.n_clusters)
        self.centers = np.array([self.freqs[self.cluster == k].mean() for k in range(self.n_clusters)])
        self.exp = extract_line_intensity(spectrum_df, self.centers, mode=self.mode, window_MHz=window_MHz)
        self.valid = np.isfinite(self.exp)

        # y of a line is a peak height for local_max/sample and an area for "area" extraction
        normalize = "area" if self.mode == "area" else "peak"
        x = spectrum_df["freq_MHz"].to_numpy(dtype=float)
        reach = window_MHz + 3.0 * float(fwhm_MHz)
        self.groups = []
        for k in range(self.n_clusters):
            c = self.centers[k]
            sel = (x >= c - window_MHz) & (x <= c + window_MHz)
            gx = x[sel]
            if gx.size == 0:
                continue
            near = np.where(np.abs(self.freqs - c) <= reach)[0]
            P = lineshape_values(gx[:, None], self.freqs[near][None, :], fwhm_MHz, profile=profile, normalize=normalize)
            self.groups.append({"k": k, "idx": near, "gx": gx, "P": P, "i_sample": int(np.argmin(np.abs(gx - c)))})

    def apply(self, y_lines: np.ndarray) -> np.ndarray:
        """Model value of every cluster for the model intensities ``y_lines`` (one per transition)."""
        y_lines = np.asarray(y_lines, dtype=float)
        out = np.full(self.n_clusters, np.nan)
        for g in self.groups:
            s = g["P"] @ y_lines[g["idx"]]
            if self.mode == "local_max":
                out[g["k"]] = s.max()
            elif self.mode == "sample":
                out[g["k"]] = s[g["i_sample"]]
            else:
                out[g["k"]] = integrate_area(s, g["gx"])
        return out
