# -*- coding: utf-8 -*-
"""fig2_rice_pcc.py — Rice (indica, control): Kode vs 作者两变体
风格: PCC = 线+ribbon(pooled SD); MSE = log10 分组箱线图
线型: recursive = 实线, iterative = 虚线
面板: (a) recursive PCC  (b) iterative PCC  (c) recursive MSE  (d) iterative MSE

数据 (同目录):  unified_norm_author_vs_kode_per_timepoint.csv
                unified_norm_author_vs_kode_per_foldset.csv
输出 (同目录):  fig2_rice_pcc.svg
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
CSV   = os.path.join(HERE, "unified_norm_author_vs_kode_per_timepoint_v2.csv")
LONG  = os.path.join(HERE, "unified_norm_author_vs_kode_per_foldset_v2.csv")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "axes.linewidth": 0.85, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "legend.fontsize": 12, "savefig.dpi": 300, "figure.dpi": 110,
    "svg.fonttype": "none",
})

METHODS = [("Kode",       "#D62728", "Kode"),
           ("author_tp1", "#1F77B4", "DynamicGP-MegaLMM+TP1"),
           ("author_mega","#2CA02C", "DynamicGP-MegaLMM"),
           ("author_rr",  "#444444", "DynamicGP-RR-BLUP")]
DODGE = {m[0]: off for m, off in zip(METHODS, [-0.27, -0.09, 0.09, 0.27])}


def style_axis(ax, title, ylab):
    ax.set_ylabel(ylab, fontweight="bold", fontsize=13.5)
    ax.set_title(title, fontsize=16, fontweight="bold", loc="left", pad=6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=3)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")


def panel_label(ax, lab):
    ax.text(-0.13, 1.04, lab, transform=ax.transAxes, fontsize=23, fontweight="bold")


def trait_sd(long, das):
    """跨性状 SD (对齐作者第一篇 "mean across all traits +/- SD"): 每个性状先跨折平均, 再取性状间 SD. 逐 DAS."""
    tm = long.groupby(["Series", "DAS", "trait"])["PCC"].mean().reset_index()
    out = {}
    for s, g in tm.groupby("Series"):
        p = g.pivot(index="DAS", columns="trait", values="PCC").reindex(das)
        out[s] = p.std(axis=1).values
    return out


def draw_line(ax, d, xi, prot, metric, ls, tsd=None):
    for pfx, col, _lab in METHODS:
        y = d[f"{pfx}_{prot}_{metric}"].values.astype(float)
        s = (tsd[f"{pfx}_{prot}"] if tsd is not None else
             d[f"{pfx}_{prot}_{metric}_sd"].values.astype(float))   # 2026-09-14: 跨性状 SD
        xx = xi + DODGE[pfx]
        ax.fill_between(xx, y - s, y + s, color=col, alpha=0.20, lw=0, zorder=2)
        ax.plot(xx, y, color=col, ls=ls, lw=1.0, marker="o", ms=3.0,
                mec=col, mfc=col, mew=0.25, zorder=3)


def draw_box(ax, long, das, prot, ls):
    data, pos, cols = [], [], []
    for i, tp in enumerate(das):
        for pfx, col, _lab in METHODS:
            v = long[(long.Series == f"{pfx}_{prot}") & (long.DAS == tp)]["MSE"].values
            v = v[np.isfinite(v)]
            if v.size == 0:
                v = np.array([np.nan])
            data.append(np.log10(v + 1e-12))
            pos.append(float(tp) + DODGE[pfx]); cols.append(col)   # 真实时间位置(与折线一致)
    bp = ax.boxplot(data, positions=pos, widths=0.17, patch_artist=True,
                    showfliers=True, manage_ticks=False,
                    flierprops=dict(marker="o", markersize=0.8, markerfacecolor="0.35",
                                    markeredgecolor="none", alpha=0.15),
                    whiskerprops=dict(linewidth=0.35, color="0.25", linestyle=ls),
                    capprops=dict(linewidth=0.35, color="0.25", linestyle=ls),
                    boxprops=dict(linewidth=0.35, linestyle=ls),
                    medianprops=dict(linewidth=0.6, color="0.1", linestyle=ls))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_edgecolor(c); patch.set_alpha(0.80)
    ax.set_ylim(-5.0, -1.0); ax.set_yticks([-5, -4, -3, -2, -1])


def main():
    d    = pd.read_csv(CSV)
    long = pd.read_csv(LONG)
    das  = list(d["DAS"].values)
    xi   = np.asarray(das, dtype=float)   # 真实时间位置(非索引)

    fig, axes = plt.subplots(4, 1, figsize=(8.2, 15.2))

    TSD = trait_sd(long, das)
    draw_line(axes[0], d, xi, "rec",  "PCC", "-", tsd=TSD)
    style_axis(axes[0], "Rice (indica, control): recursive — PCC", "Prediction accuracy (r)")
    panel_label(axes[0], "a")
    axes[0].set_ylim(0.4, 1.0); axes[0].set_yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0])

    draw_line(axes[1], d, xi, "iter", "PCC", "--", tsd=TSD)
    style_axis(axes[1], "Rice (indica, control): iterative — PCC", "Prediction accuracy (r)")
    panel_label(axes[1], "b")
    axes[1].set_ylim(0.4, 1.0); axes[1].set_yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0])

    draw_box(axes[2], long, das, "rec", "-")
    style_axis(axes[2], "Rice (indica, control): recursive — MSE", r"$\log_{10}$(MSE)")
    panel_label(axes[2], "c")

    draw_box(axes[3], long, das, "iter", "--")
    style_axis(axes[3], "Rice (indica, control): iterative — MSE", r"$\log_{10}$(MSE)")
    panel_label(axes[3], "d")

    # 四个面板统一使用同一套 DAP 刻度(否则前三个面板会显示 0/2/4... 的索引刻度)
    for ax in axes:
        ax.set_xlim(xi[0] - 0.6, xi[-1] + 0.6)   # 四面板统一真实时间范围, 保证刻度/点位对齐
        ax.set_xticks(xi)
        ax.set_xticklabels([str(int(v)) for v in das], rotation=45, ha="right",
                           fontweight="bold")
    axes[-1].set_xlabel("DAS", fontweight="bold", fontsize=13.5)

    handles = [Patch(facecolor=c, edgecolor=c, alpha=0.80, label=lab)
               for _p, c, lab in METHODS]
    leg = fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
                     title="Method", bbox_to_anchor=(0.5, 1.0),
                     handlelength=1.6, columnspacing=2.4, borderaxespad=0.2)
    leg.get_title().set_fontweight("bold")
    for t in leg.get_texts():
        t.set_fontweight("bold")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    for ext in ("svg",):   # 只输出 SVG
        out = os.path.join(HERE, f"fig2_rice_pcc_v2.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("saved ->", out)
    plt.close(fig)


if __name__ == "__main__":
    print("data  ->", CSV)
    print("long  ->", LONG)
    main()
