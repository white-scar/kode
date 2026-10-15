# -*- coding: utf-8 -*-
"""fig_rice_inwindow.py - rice (indica control) in-window: Kode vs dynamicGP-MegaLMM+TP1.
Same train-fold per-trait min-max space on both sides -> MSE directly comparable.
input timepoint DAS28 excluded (it is the given true t1)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.linewidth": 0.8, "xtick.labelsize": 7.6, "ytick.labelsize": 8.2,
    "legend.fontsize": 7.4, "savefig.dpi": 300, "figure.dpi": 110, "svg.fonttype": "none",
})
D = pd.read_csv(f"{HERE}/rice_control_bytime.csv")
D = D[D.timepoint != 28]                       # DAS28 = given true input
DAS = sorted(D.timepoint.unique()); x = np.arange(len(DAS), dtype=float)

STYLE = [
    ("Kode", "rec",  "#D62728", "-",  1.8),
    ("Kode", "iter", "#D62728", "--", 1.3),
    ("dynamicGP-MegaLMM+TP1", "rec",  "#1F77B4", "-",  1.8),
    ("dynamicGP-MegaLMM+TP1", "iter", "#1F77B4", "--", 1.3),
]
def get(meth, scen, col):
    d = D[(D.method == meth) & (D.scenario == scen)].set_index("timepoint").reindex(DAS)
    return d[col].values

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax, col, sd, ylab, title, logy in [
        (axes[0], "pcc", "pcc_sd", "Prediction accuracy (r)", "PCC", False),
        (axes[1], "MSE_norm", "mse_sd", "MSE (normalized)", "MSE", True)]:
    for meth, scen, c, ls, lw in STYLE:
        mu = get(meth, scen, col); s = get(meth, scen, sd)
        lo, hi = mu - s, mu + s
        if logy:
            mu = np.clip(mu, 1e-6, None); lo = np.clip(lo, 1e-6, None)
        ax.fill_between(x, lo, hi, color=c, alpha=0.16, lw=0, zorder=2)
        ax.plot(x, mu, color=c, lw=lw, ls=ls, zorder=3)
        ax.plot(x, mu, "o", color=c, ms=3.0, mec="white", mew=0.4, zorder=4)
    if logy:
        ax.set_yscale("log"); ax.set_ylim(1e-3, 2e-1)
    else:
        ax.set_ylim(0.45, 1.0); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticks(x); ax.set_xticklabels([str(d) for d in DAS], rotation=45, ha="right")
    ax.set_xlabel("DAS", fontweight="bold"); ax.set_ylabel(ylab, fontweight="bold")
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", pad=6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=3)
    for lab in ax.get_xticklabels() + ax.get_yticklabels(): lab.set_fontweight("bold")
    ax.text(-0.17, 1.03, "a" if ax is axes[0] else "b", transform=ax.transAxes,
            fontsize=15, fontweight="bold")

handles = [Line2D([], [], color=c, lw=lw, ls=ls, marker="o", ms=3.0, mec="white", mew=0.4)
           for _, _, c, ls, lw in STYLE]
labels = ["Kode  recursive", "Kode  iterative",
          "dynamicGP-MegaLMM+TP1  recursive", "dynamicGP-MegaLMM+TP1  iterative"]
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.05), handlelength=1.9, columnspacing=1.5, labelspacing=0.4)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(f"{HERE}/fig_rice_inwindow.svg", format="svg")
fig.savefig(os.path.join(os.environ.get("TEMP", "/tmp"), "_rice_check.png"), dpi=160)
print("saved fig_rice_inwindow.svg")
