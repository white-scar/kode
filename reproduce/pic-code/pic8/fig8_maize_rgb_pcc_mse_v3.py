# -*- coding: utf-8 -*-
"""Figure 8 | fig8_maize_rgb_pcc_mse.py -- Maize UAV RGB (Adak et al. 2023): Kode vs baselines

Style follows pic-code/pic2/fig2_maize_pcc.py
  PCC : line + ribbon = mean +/- pooled SD (over trait x fold)   [per_timepoint csv]
  MSE : grouped boxplot of log10(MSE), each box pools (fold x trait) = 170 values
  x   : REAL DAS spacing (days), not equal-spaced categories
Panels: (a) drought PCC (b) drought MSE  [v2: irrigated 未跑, 故只有两面板]

Data (same folder): maize_rgb_per_timepoint.csv, maize_rgb_per_foldset.csv
Output: fig8_maize_rgb_pcc_mse_v3.svg   (Figure 8)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
TAB  = os.path.join(HERE, "maize_rgb_per_timepoint_v2.csv")
FOLD = os.path.join(HERE, "maize_rgb_per_foldset_v2.csv")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "font.size": 10.5, "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.linewidth": 0.8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 10.5, "savefig.dpi": 300, "figure.dpi": 110,
    "svg.fonttype": "none",
})

# Series key, colour, legend label, line style
METHODS = [("Kode_rec",         "#D62728", "Kode (recursive)",  "-"),
           ("Kode_iter",        "#D62728", "Kode (iterative)", "--"),
           ("baseline_rrBLUPt1", "#444444", "baseline", ":"),
           ("baseline_t1",      "#999999", "persistence t1", "-.")]
DODGE = {m[0]: off for m, off in zip(METHODS, [-0.45, -0.15, 0.15, 0.45])}   # DAS days
W_BOX = 0.22   # box width in days (narrow, like pic5)
ENVS = [("drought", "Drought"), ("irrigated", "Irrigated")]
FLOWERING = (61, 82)   # heat-associated flowering period (Adak et al. 2023, Fig. 10A)
ACUTE     = (68, 75)   # flights with acute trajectory deviation


def style_axis(ax, title, ylab):
    ax.set_ylabel(ylab, fontweight="bold", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=3)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")


def panel_label(ax, lab):
    ax.text(-0.13, 1.04, lab, transform=ax.transAxes, fontsize=20, fontweight="bold")


def shade_event(ax, annotate=False):
    y0, y1 = ax.get_ylim()
    # (i) heat-associated flowering period reported by the source study (61-82 DAP)
    ax.axvspan(FLOWERING[0] - 1.5, FLOWERING[1] + 1.5, color="0.5", alpha=0.07, lw=0, zorder=0)
    # (ii) flights at which the trajectory deviates from the smooth developmental trend
    ax.axvspan(ACUTE[0] - 1.5, ACUTE[1] + 1.5, color="0.5", alpha=0.17, lw=0, zorder=0)
    if annotate:
        ax.text((FLOWERING[0] + FLOWERING[1]) / 2.0, y1,
                "heat-associated flowering period (Adak et al. 2023)\n"
                "flights with acute deviation: DAP 68, 71, 75",
                ha="center", va="top", fontsize=6.8, color="0.28", style="italic",
                linespacing=1.4, zorder=6)


def trait_sd(fold, das, env):
    """跨性状 SD: 每个性状先跨折平均, 再取性状间 SD (逐 DAS)."""
    sub = fold[fold.Series.str.endswith("_" + env)]
    tm = sub.groupby(["Series", "DAS", "trait"])["PCC"].mean().reset_index()
    out = {}
    for s_, g in tm.groupby("Series"):
        p_ = g.pivot(index="DAS", columns="trait", values="PCC").reindex(das)
        out[s_] = p_.std(axis=1).values
    return out


def draw_line(ax, tab, das, env, metric, tsd=None):
    xi = np.asarray(das, dtype=float)
    for key, col, _lab, ls in METHODS:
        y = tab[f"{key}_{env}_{metric}"].values.astype(float)
        s = (tsd[f"{key}_{env}"] if tsd is not None else
             tab[f"{key}_{env}_{metric}_sd"].values.astype(float))   # 跨性状 SD
        ax.fill_between(xi, y - s, y + s, color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(xi, y, color=col, ls=ls, lw=(2.4 if col == "#D62728" else 1.7),
                marker="o", ms=(4.6 if col == "#D62728" else 3.4),
                mec=col, mfc=col, mew=0.3, zorder=(5 if col == "#D62728" else 3))


def draw_box(ax, fold, das, env):
    sub = fold[fold.Series.str.endswith("_" + env)]
    data, pos, cols, lss = [], [], [], []
    for dasv in das:
        for key, col, _lab, _ls in METHODS:
            v = sub[(sub.Series == f"{key}_{env}") & (sub.DAS == dasv)]["MSE"].dropna().values
            v = v[np.isfinite(v) & (v > 0)]
            if v.size == 0:
                v = np.array([np.nan])
            data.append(np.log10(v + 1e-12))
            pos.append(float(dasv) + DODGE[key]); cols.append(col); lss.append(_ls)
    bp = ax.boxplot(data, positions=pos, widths=W_BOX, patch_artist=True,
                    showfliers=False, manage_ticks=False,
                    whiskerprops=dict(linewidth=0.45, color="0.3"),
                    capprops=dict(linewidth=0.45, color="0.3"),
                    boxprops=dict(linewidth=0.7),
                    medianprops=dict(linewidth=1.1, color="0.05"))
    for patch, c, ls in zip(bp["boxes"], cols, lss):
        patch.set_facecolor(c); patch.set_edgecolor(c)
        patch.set_alpha(0.90); patch.set_linestyle(ls)
    ls2 = [x for x in lss for _ in range(2)]      # 每个箱有上下两根 whisker/cap
    for el, ls in zip(bp["whiskers"], ls2): el.set_linestyle(ls)
    for el, ls in zip(bp["caps"],     ls2): el.set_linestyle(ls)
    for el, ls in zip(bp["medians"],  lss): el.set_linestyle(ls); el.set_color("0.05")


def main():
    tab = pd.read_csv(TAB)
    fold = pd.read_csv(FOLD)
    das = list(tab.DAS.values)
    xi = np.asarray(das, dtype=float)
    print("per_timepoint:", tab.shape, " per_foldset:", fold.shape)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.6))     # v2: 只做 drought (irrigated 新流程未跑)
    rows = [("drought", "PCC", "a"), ("drought", "MSE", "b")]
    for ax, (env, metric, lab) in zip(axes, rows):
        envlab = dict(ENVS)[env]
        if metric == "PCC":
            draw_line(ax, tab, das, env, "PCC", tsd=trait_sd(fold, das, env))
            style_axis(ax, f"Maize UAV RGB ({envlab}): Kode vs baselines -- PCC",
                       "Prediction accuracy (r)")
            ax.set_ylim(-0.1, 1.05); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            shade_event(ax, annotate=True)
        else:
            draw_box(ax, fold, das, env)
            style_axis(ax, f"Maize UAV RGB ({envlab}): Kode vs baselines -- MSE",
                       r"$\log_{10}$(MSE)")
            ax.set_ylim(-4.0, 0.5); ax.set_yticks([-4, -3, -2, -1, 0])
            shade_event(ax, annotate=True)
        panel_label(ax, lab)

    for ax in axes[:-1]:
        ax.set_xticks(xi); ax.set_xlim(min(das) - 3, max(das) + 3)
        ax.tick_params(labelbottom=False)
    axes[-1].set_xticks(xi)
    axes[-1].set_xticklabels([str(int(v)) for v in das], rotation=45, ha="right", fontweight="bold")
    axes[-1].set_xlim(min(das) - 3, max(das) + 3)
    axes[-1].set_xlabel("DAS", fontweight="bold", fontsize=12)

    handles = [Patch(facecolor=c, edgecolor=c, alpha=0.80, label=l) for _k, c, l, _ls in METHODS]
    leg = fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                     title="Method", bbox_to_anchor=(0.5, 1.0),
                     handlelength=1.6, columnspacing=1.5, borderaxespad=0.2)
    leg.get_title().set_fontweight("bold")
    for t in leg.get_texts():
        t.set_fontweight("bold")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = os.path.join(HERE, "fig8_maize_rgb_pcc_mse_v3.svg")
    fig.savefig(out, bbox_inches="tight")
    print("saved ->", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
