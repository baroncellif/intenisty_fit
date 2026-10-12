from __future__ import annotations

import numpy as np


def lineshape_values(x, center: float, fwhm: float, profile: str = "gaussian", normalize: str = "area"):
    x = np.asarray(x, dtype=float)
    fwhm = max(float(fwhm), np.finfo(float).eps)
    profile = str(profile).lower()

    if profile == "gaussian":
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        y = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        area = sigma * np.sqrt(2.0 * np.pi)
    elif profile == "lorentzian":
        gamma = 0.5 * fwhm
        y = gamma**2 / ((x - center) ** 2 + gamma**2)
        area = np.pi * gamma
    else:
        raise ValueError(f"Unknown line-shape profile: {profile}")

    if normalize == "area":
        return y / area
    if normalize == "peak":
        return y
    raise ValueError(f"Unknown line-shape normalization: {normalize}")


def render_profiles(x, centers, intensities, fwhm: float, profile: str = "gaussian", normalize: str = "area"):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    for center, intensity in zip(np.asarray(centers, dtype=float), np.asarray(intensities, dtype=float)):
        y += intensity * lineshape_values(x, center, fwhm, profile=profile, normalize=normalize)
    return y


def lineshape_options(fit_options) -> dict:
    opts = dict(fit_options.lineshape or {})
    return {
        "profile": opts.get("profile", "gaussian"),
        "fwhm_MHz": float(opts.get("fwhm_MHz", 0.08)),
        "fit_fwhm": bool(opts.get("fit_fwhm", False)),
        "fwhm_bounds_MHz": opts.get("fwhm_bounds_MHz", [0.005, 1.0]),
        "window_MHz": float(opts.get("window_MHz", fit_options.extraction_window_MHz)),
        "normalize": opts.get("normalize", "area"),
    }
