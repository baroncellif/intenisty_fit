from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .lineshape import lineshape_options, render_profiles
from .model import intensity_at_T_approx, intensity_at_T_exact


def gaussian_broadening(x, centers, intensities, fwhm_MHz: float = 0.03):
    sigma = fwhm_MHz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    y = np.zeros_like(x, dtype=float)
    for c, a in zip(np.asarray(centers, dtype=float), np.asarray(intensities, dtype=float)):
        y += a * np.exp(-0.5 * ((x - c) / sigma) ** 2)
    return y


def build_prediction_table_for_plot(species_tables, result, fit_options, q_functions=None, use_lin_frequency=True):
    q_functions = q_functions or {}
    rows = []
    for name, tab in species_tables.items():
        Tnow = result["species"][name]["Trot_K"]
        scale = result["species"][name]["scale"]
        I_cat = tab["I_cat"].to_numpy(dtype=float)
        Elo = tab["elo_cm-1"].to_numpy(dtype=float)
        DR = tab["dr"].to_numpy(dtype=float)
        nu = tab["cat_freq_MHz"].to_numpy(dtype=float)
        Tcat = tab["cat_temperature_K"].to_numpy(dtype=float)
        if fit_options.use_exact_temperature_scaling and name in q_functions:
            Icat_T = intensity_at_T_exact(I_cat, Elo, nu, Tnow, Tcat, q_functions[name])
        else:
            Icat_T = intensity_at_T_approx(I_cat, Elo, DR, Tnow, Tcat)
        Imodel = scale * Icat_T
        freq_col = "lin_freq_MHz" if use_lin_frequency else "cat_freq_MHz"
        for i in range(len(tab)):
            rows.append({
                "species": name,
                "freq_MHz": float(tab.iloc[i][freq_col]),
                "pred_intensity": float(Imodel[i]),
                "dipole_component": tab.iloc[i].get("dipole_component", "unknown"),
            })
    return pd.DataFrame(rows)


def write_quick_overlay_plot(path, spectrum_df, line_diag_df, fit_options, result=None, fmin=None, fmax=None):
    x, y_exp, y_fit, fmin, fmax = _overlay_arrays(
        spectrum_df,
        line_diag_df,
        fit_options,
        result=result,
        fmin=fmin,
        fmax=fmax,
    )
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, y_exp, color="black", lw=0.8, label="exp")
    ax.plot(x, y_fit, color="red", lw=1.0, label="fit")
    ax.set_xlabel("Frequency / MHz")
    ax.set_ylabel("Intensity")
    ax.set_xlim(fmin, fmax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def show_interactive_overlay_plot(spectrum_df, line_diag_df, fit_options, result=None, fmin=None, fmax=None):
    if line_diag_df.empty:
        return

    plot_cfg = dict(fit_options.plot or {})
    lopts = lineshape_options(fit_options)
    fwhm = result_fwhm(result, fit_options)

    if fmin is None:
        fmin = float(line_diag_df["lin_freq_MHz"].min() - 1.0)
    if fmax is None:
        fmax = float(line_diag_df["lin_freq_MHz"].max() + 1.0)

    spec_win = spectrum_df[
        (spectrum_df["freq_MHz"] >= fmin) &
        (spectrum_df["freq_MHz"] <= fmax)
    ].copy()
    pred_win = line_diag_df[
        (line_diag_df["lin_freq_MHz"] >= fmin) &
        (line_diag_df["lin_freq_MHz"] <= fmax)
    ].copy()

    species_to_plot = plot_cfg.get("species_to_plot")
    if species_to_plot is not None:
        pred_win = pred_win[pred_win["species"].isin(species_to_plot)].copy()
    color_by = str(plot_cfg.get("interactive_color_by", "species_component")).lower()
    if color_by == "species":
        pred_win["plot_group"] = pred_win["species"].astype(str)
    elif color_by == "component":
        pred_win["plot_group"] = pred_win["dipole_component"].astype(str)
    elif color_by == "species_component":
        pred_win["plot_group"] = pred_win["species"].astype(str) + ":" + pred_win["dipole_component"].astype(str)
    else:
        raise ValueError("plot.interactive_color_by must be 'species', 'component', or 'species_component'")

    xexp = spec_win["freq_MHz"].to_numpy(dtype=float)
    yexp = spec_win["intensity"].to_numpy(dtype=float)

    if bool(plot_cfg.get("normalize_exp", True)) and yexp.size and np.nanmax(np.abs(yexp)) > 0:
        yexp = yexp / np.nanmax(np.abs(yexp))

    mode = str(plot_cfg.get("interactive_mode", "sticks")).lower()
    npoints = int(plot_cfg.get("interactive_npoints", plot_cfg.get("npoints", 100000)))
    npoints = max(npoints, 1000)
    show_total = bool(plot_cfg.get("show_total", False))
    total_on_top = bool(plot_cfg.get("total_on_top", False))

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=tuple(plot_cfg.get("interactive_figsize", (12, 7))),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    ax1.plot(xexp, yexp, color="black", lw=1.0)
    ax1.set_ylabel("Exp.")
    ax1.set_xlim(fmin, fmax)

    if not pred_win.empty:
        group_list = list(pred_win["plot_group"].drop_duplicates())
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        xgrid = np.linspace(fmin, fmax, npoints)
        total_y = np.zeros_like(xgrid)
        plot_items = []

        ax2.axhline(0.0, color="0.35", lw=0.8)

        for i, group in enumerate(group_list):
            sub = pred_win[pred_win["plot_group"] == group]
            xc = sub["lin_freq_MHz"].to_numpy(dtype=float)
            yc = sub["model_intensity"].to_numpy(dtype=float)
            color = colors[i % len(colors)]

            if mode == "sticks":
                plot_items.append(("sticks", xc, yc, color, group))
                if show_total:
                    total_y += render_profiles(
                        xgrid,
                        xc,
                        yc,
                        fwhm=fwhm,
                        profile=lopts["profile"],
                        normalize=lopts["normalize"],
                    )
            elif mode == "broadened":
                yb = render_profiles(
                    xgrid,
                    xc,
                    yc,
                    fwhm=fwhm,
                    profile=lopts["profile"],
                    normalize=lopts["normalize"],
                )
                plot_items.append(("line", xgrid, yb, color, group))
                total_y += yb
            else:
                raise ValueError("plot.interactive_mode must be 'sticks' or 'broadened'")

        if show_total:
            plot_items.append(("total", xgrid, total_y, None, "total"))

        pred_scale = 1.0
        if bool(plot_cfg.get("normalize_pred", True)):
            maxima = []
            for _kind, _x, yvals, _color, _label in plot_items:
                if len(yvals):
                    ymax = np.nanmax(np.abs(yvals))
                    if np.isfinite(ymax):
                        maxima.append(ymax)
            if maxima and max(maxima) > 0:
                pred_scale = max(maxima)

        for kind, xvals, yvals, color, label in plot_items:
            yplot = yvals / pred_scale
            if kind == "sticks":
                ax2.vlines(xvals, 0.0, yplot, lw=1.2, alpha=0.9, color=color, label=label)
            elif kind == "line":
                ax2.plot(xvals, yplot, lw=1.0, alpha=0.9, color=color, label=label)
            elif kind == "total":
                ax2.plot(
                    xvals,
                    yplot,
                    lw=1.5 if total_on_top else 1.2,
                    ls="--",
                    label=label,
                    alpha=1.0 if total_on_top else 0.7,
                    zorder=20 if total_on_top else 1,
                )

        if show_total:
            total_y = total_y / pred_scale

        ax2.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            fontsize=8,
            ncol=1,
            borderaxespad=0.0,
        )

    ax2.set_xlabel("Frequency / MHz")
    ax2.set_ylabel("Pred.")
    ax1.set_xlim(fmin, fmax)
    ax2.set_xlim(fmin, fmax)

    plt.ion()
    plt.show(block=True)


def _overlay_arrays(spectrum_df, line_diag_df, fit_options, result=None, fmin=None, fmax=None):
    if line_diag_df.empty:
        return np.array([]), np.array([]), np.array([]), fmin, fmax
    lopts = lineshape_options(fit_options)
    fwhm = result_fwhm(result, fit_options)
    if fmin is None:
        fmin = float(line_diag_df["lin_freq_MHz"].min() - 1.0)
    if fmax is None:
        fmax = float(line_diag_df["lin_freq_MHz"].max() + 1.0)

    spec = spectrum_df[(spectrum_df["freq_MHz"] >= fmin) & (spectrum_df["freq_MHz"] <= fmax)]
    x = spec["freq_MHz"].to_numpy(dtype=float)
    y_exp = spec["intensity"].to_numpy(dtype=float)
    sub = line_diag_df[(line_diag_df["lin_freq_MHz"] >= fmin) & (line_diag_df["lin_freq_MHz"] <= fmax)]
    y_fit = render_profiles(
        x,
        sub["lin_freq_MHz"].to_numpy(dtype=float),
        sub["model_intensity"].to_numpy(dtype=float),
        fwhm=fwhm,
        profile=lopts["profile"],
        normalize=lopts["normalize"],
    )
    plot_cfg = dict(fit_options.plot or {})
    normalize_exp = bool(plot_cfg.get("normalize_exp", True))
    normalize_pred = bool(plot_cfg.get("normalize_pred", True))
    if normalize_exp and y_exp.size and np.nanmax(np.abs(y_exp)) > 0:
        y_exp = y_exp / np.nanmax(np.abs(y_exp))
    if normalize_pred and y_fit.size and np.nanmax(np.abs(y_fit)) > 0:
        y_fit = y_fit / np.nanmax(np.abs(y_fit))
    return x, y_exp, y_fit, fmin, fmax


def write_species_line_pdf(path, spectrum_df, line_diag_df, fit_options, result=None, lines_per_page=25, ncols=5):
    if line_diag_df.empty:
        return
    lopts = lineshape_options(fit_options)
    plot_cfg = dict(fit_options.plot or {})
    fwhm = result_fwhm(result, fit_options)
    window = float(plot_cfg.get("line_window_MHz", max(5.0 * fwhm, lopts["window_MHz"])))
    ncols = int(plot_cfg.get("ncols", ncols))
    lines_per_page = int(plot_cfg.get("lines_per_page", lines_per_page))
    normalize_panels = bool(plot_cfg.get("normalize_panels", True))
    nrows = int(np.ceil(lines_per_page / ncols))

    with PdfPages(path) as pdf:
        for species, sp_df in line_diag_df.groupby("species", sort=False):
            sp_df = sp_df.sort_values("lin_freq_MHz").reset_index(drop=True)
            for start in range(0, len(sp_df), lines_per_page):
                page = sp_df.iloc[start:start + lines_per_page]
                fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.15 * nrows), squeeze=False)
                fig.suptitle(f"{species} lines {start + 1}-{start + len(page)}", fontsize=11)
                for ax in axes.ravel():
                    ax.axis("off")

                for ax, (_, row) in zip(axes.ravel(), page.iterrows()):
                    ax.axis("on")
                    center = float(row["lin_freq_MHz"])
                    mask = (spectrum_df["freq_MHz"] >= center - window) & (spectrum_df["freq_MHz"] <= center + window)
                    spec = spectrum_df.loc[mask]
                    if spec.empty:
                        continue
                    x = spec["freq_MHz"].to_numpy(dtype=float)
                    y_exp = spec["intensity"].to_numpy(dtype=float)

                    nearby = line_diag_df[
                        (line_diag_df["lin_freq_MHz"] >= center - window) &
                        (line_diag_df["lin_freq_MHz"] <= center + window)
                    ]
                    y_total = render_profiles(
                        x,
                        nearby["lin_freq_MHz"].to_numpy(dtype=float),
                        nearby["model_intensity"].to_numpy(dtype=float),
                        fwhm=fwhm,
                        profile=lopts["profile"],
                        normalize=lopts["normalize"],
                    )
                    y_line = render_profiles(
                        x,
                        [center],
                        [float(row["model_intensity"])],
                        fwhm=fwhm,
                        profile=lopts["profile"],
                        normalize=lopts["normalize"],
                    )
                    if normalize_panels:
                        exp_scale = np.nanmax(np.abs(y_exp)) if y_exp.size else np.nan
                        fit_scale = np.nanmax(np.abs(y_total)) if y_total.size else np.nan
                        if np.isfinite(exp_scale) and exp_scale > 0:
                            y_exp = y_exp / exp_scale
                        if np.isfinite(fit_scale) and fit_scale > 0:
                            y_total = y_total / fit_scale
                            y_line = y_line / fit_scale

                    ax.plot(x, y_exp, color="black", lw=0.8)
                    ax.plot(x, y_total, color="red", lw=0.9)
                    ax.plot(x, y_line, color="blue", lw=0.8)
                    ax.axvline(center, color="0.5", lw=0.5, alpha=0.5)
                    ax.set_xlim(center - window, center + window)
                    ax.tick_params(labelsize=6, length=2)
                    ax.set_title(f"{center:.3f} MHz {row.get('dipole_component', '')}", fontsize=7)
                    ax.set_xlabel("MHz", fontsize=6)
                    ax.set_ylabel("I" + (" norm." if normalize_panels else ""), fontsize=6)

                fig.tight_layout(rect=[0, 0, 1, 0.97])
                pdf.savefig(fig)
                plt.close(fig)


def result_fwhm(result, fit_options):
    if result is not None:
        val = result.get("fit_stats", {}).get("lineshape_fwhm_MHz")
        if val is not None and np.isfinite(val):
            return float(val)
    return float(lineshape_options(fit_options)["fwhm_MHz"])
