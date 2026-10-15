# -*- coding: utf-8 -*-
"""fig7_extrap_maize.py — 玉米外推 (forecasting) 主图

a  协议示意: 训练 t1-t20 (DAS15-40) / 预测 t21-t25 (DAS43-47) / 真 t20 作 forecast anchor
b  matched-CV3 forecasting PCC   —— 线 + mean±SD ribbon (SD 跨 10 fold-set)
c  matched-CV3 forecasting MSE   —— log10 分组箱线 (逐 trait, 每箱 n = 10 fold-set x 50 traits)
d  三种信息条件总体对照 (paired dot): t1-only / t20-anchored / iterative, 左 PCC 右 MSE, 误差棒 ±SD

数据 (同目录):
  fig7b_cv3_traj_pcc_mse.csv     逐 DAS 的 mean/sd (rec_tk, 两方法)
  fig7c_cv3_bytrait_mse.csv      逐 trait 的 MSE (rec_tk, 两方法)
  fig7d_overall_scenarios.csv    三条件的总体 mean/sd
输出: fig7_extrap_maize.svg (只输出 SVG)
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "font.size": 10, "axes.titlesize": 11.5, "axes.labelsize": 10,
    "axes.linewidth": 0.8, "xtick.labelsize": 8.8, "ytick.labelsize": 8.8,
    "legend.fontsize": 8.8, "savefig.dpi": 300, "figure.dpi": 110, "svg.fonttype": "none",
})
C_K, C_D = "#D62728", "#1F77B4"                      # Kode 红 / dynamicGP 蓝
LAB_K, LAB_D = "Kode", "dynamicGP-MegaLMM+TP1"
FC = [43, 44, 45, 46, 47]

B = pd.read_csv(f"{HERE}/fig7b_cv3_traj_pcc_mse.csv")
C = pd.read_csv(f"{HERE}/fig7c_cv3_bytrait_mse.csv")
Dd = pd.read_csv(f"{HERE}/fig7d_overall_scenarios.csv")

def style(ax, title, ylab=None, xlab=None):
    if ylab: ax.set_ylabel(ylab, fontweight="bold", fontsize=10)
    if xlab: ax.set_xlabel(xlab, fontweight="bold", fontsize=10)
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left", pad=6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=3)
    for lab in ax.get_xticklabels() + ax.get_yticklabels(): lab.set_fontweight("bold")

def plabel(ax, lab, dx=-0.097, dy=0.008):
    """面板字母: 用图坐标的同一个 x 放置, 保证 a/b/c/d 左对齐 (各行轴宽度不同, 不能用 transAxes 的相对偏移)."""
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, lab, fontsize=17, fontweight="bold")

fig = plt.figure(figsize=(7.0, 11.0))
gs = fig.add_gridspec(4, 2, height_ratios=[0.82, 1.0, 1.15, 1.0], hspace=0.66, wspace=0.28,
                      left=0.135, right=0.985, top=0.93, bottom=0.065)   # 2026-09-14: 左边界右移, 避免 y 轴标题(含 mathtext)落在 abcd 左侧

# ---------- a) 协议示意 ----------
axa = fig.add_subplot(gs[0, :]); axa.set_xlim(-0.6, 25.4); axa.set_ylim(-1.5, 2.5); axa.axis("off")
axa.add_patch(Rectangle((-0.5, 0.05), 20.0, 0.9, facecolor="#EAF1FA", edgecolor="none", zorder=0))
axa.add_patch(Rectangle((19.5, 0.05), 5.6, 0.9, facecolor="#FBE9E7", edgecolor="none", zorder=0))
axa.text(9.6, 1.35, "training window  t1-t20  (DAS15-40)", ha="center", fontsize=8.4, color="#2C5D9E", fontweight="bold")
axa.text(22.3, 1.35, "forecast horizon  t21-t25  (DAS43-47)", ha="center", fontsize=8.4, color="#B23A2E", fontweight="bold")
for i in range(25):
    axa.plot([i+0.5], [0.5], marker="o", ms=3.6 if i < 20 else 3.0,
             mfc="#3C6DB0" if i < 20 else "white", mec="#3C6DB0", mew=0.9, zorder=3)
axa.plot([19.5], [0.5], marker="o", ms=7.0, mfc=C_K, mec="white", mew=0.8, zorder=5)
axa.add_patch(FancyArrowPatch((19.5, -0.35), (19.5, 0.28), arrowstyle="-|>", mutation_scale=7, lw=1.0, color=C_K, zorder=5))
axa.text(19.5, -0.62, "true t20\n(forecast anchor)", ha="center", va="top", fontsize=7.6, color=C_K, fontweight="bold")
axa.text(22.5, -0.62, "5 unseen timepoints\n(all methods)", ha="center", va="top", fontsize=7.6, color="#B23A2E")
axa.text(-0.4, -1.25, "other settings:  t1-only rollout (autonomous)   ·   iterative (observation-assisted)",
         ha="left", va="top", fontsize=7.4, color="0.35", style="italic")

# ---------- b) PCC (线 + ribbon) ----------
axb = fig.add_subplot(gs[1, :])
for who, col, lab in ((LAB_D, C_D, LAB_D), (LAB_K, C_K, LAB_K)):
    g = B[B.method == who].sort_values("DAS")
    axb.fill_between(g.DAS, g.pcc_mean - g.pcc_sd, g.pcc_mean + g.pcc_sd, color=col, alpha=0.20, lw=0, zorder=2)
    axb.plot(g.DAS, g.pcc_mean, color=col, lw=0.9, marker="o", ms=2.6, mec=col, mfc=col, mew=0.25, label=lab, zorder=3)
style(axb, "Matched CV3 forecasting — PCC", "Prediction accuracy (r)", "DAS")
axb.set_xticks(FC); axb.set_ylim(0.3, 1.0); axb.legend(frameon=False, loc="lower left")

# ---------- c) MSE (log10 分组箱线) ----------
axc = fig.add_subplot(gs[2, :])
for who, col, off, ls in ((LAB_D, C_D, -0.20, "--"), (LAB_K, C_K, +0.20, "-")):
    for i, t in enumerate(FC):
        v = C[(C.method == who) & (C.time == t)]["mse"].values.astype(float)
        v = np.log10(v[np.isfinite(v) & (v > 0)])
        if v.size == 0: continue
        bp = axc.boxplot([v], positions=[i + off], widths=0.15, patch_artist=True, manage_ticks=False,
                         showfliers=True,
                         flierprops=dict(marker="o", markersize=0.8, markerfacecolor="0.35", markeredgecolor="none", alpha=0.12),
                         whiskerprops=dict(linewidth=0.35, color="0.25", linestyle=ls),
                         capprops=dict(linewidth=0.35, color="0.25", linestyle=ls),
                         boxprops=dict(linewidth=0.35, linestyle=ls),
                         medianprops=dict(linewidth=0.6, color="0.1", linestyle=ls))
        bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_edgecolor(col); bp["boxes"][0].set_alpha(0.80)
style(axc, "Matched CV3 forecasting — MSE", r"$\log_{10}$(MSE)", "DAS")
axc.set_xticks(range(len(FC))); axc.set_xticklabels([str(t) for t in FC])
lg = np.log10(C.mse[C.mse > 0])
axc.set_ylim(np.floor(lg.quantile(0.01)) - 0.2, np.ceil(lg.quantile(0.99)) + 0.2)
axc.legend(handles=[Patch(facecolor=C_K, edgecolor=C_K, alpha=0.80, label=LAB_K),
                    Patch(facecolor=C_D, edgecolor=C_D, alpha=0.80, linestyle="--", label=LAB_D)],
           frameon=False, loc="lower left", fontsize=8.0, handlelength=1.5, borderpad=0.2, labelspacing=0.3)

# ---------- d) 三条件 paired dot ----------
AXP = fig.add_subplot(gs[3, 0]); AXM = fig.add_subplot(gs[3, 1])
SC = [("rec_t1", "t1-only"), ("rec_tk", "t20-anchored"), ("iter", "iterative\n(assisted)")]
ypos = np.arange(len(SC))[::-1]
def paired(ax, metric, xlab, xlim):
    for y, (s, _sl) in zip(ypos, SC):
        a = Dd[(Dd.method == LAB_D) & (Dd.scenario == s)].iloc[0]
        k = Dd[(Dd.method == LAB_K) & (Dd.scenario == s)].iloc[0]
        av, asd = a[f"{metric}_mean"], a[f"{metric}_sd"]
        kv, ksd = k[f"{metric}_mean"], k[f"{metric}_sd"]
        ax.errorbar([av], [y], xerr=[asd], fmt="o", ms=5.0, color=C_D, mec="white", mew=0.6, capsize=2.0, elinewidth=0.8, zorder=3)
        ax.errorbar([kv], [y], xerr=[ksd], fmt="o", ms=5.0, color=C_K, mec="white", mew=0.6, capsize=2.0, elinewidth=0.8, zorder=3)
        ax.text(av, y + 0.19, "%.3f" % av, ha="center", fontsize=7.0, color=C_D, fontweight="bold")
        ax.text(kv, y + 0.19, "%.3f" % kv, ha="center", fontsize=7.0, color=C_K, fontweight="bold")
    ax.set_yticks(ypos); ax.set_yticklabels([s for _k, s in SC], fontsize=8.6)
    ax.set_ylim(-0.55, len(SC) - 0.45); ax.set_xlim(*xlim)
    style(ax, xlab, None, None)
paired(AXP, "pcc", "mean PCC (t21-t25)", (0.25, 1.02))
paired(AXM, "mse", "mean MSE (t21-t25)", (0.0, 0.075))
AXP.set_title("Overall — mean PCC", fontsize=10, fontweight="bold", loc="left", pad=5)
AXM.set_title("Overall — mean MSE", fontsize=10, fontweight="bold", loc="left", pad=5)
AXM.set_yticklabels([])
AXP.legend(handles=[Line2D([], [], marker="o", ls="", ms=5.0, color=C_D, mec="white", label=LAB_D),
                    Line2D([], [], marker="o", ls="", ms=5.0, color=C_K, mec="white", label=LAB_K)],
           frameon=False, loc="upper right", fontsize=7.6)

# ---------- 统一面板字母: 放在全部文本墨迹的最左侧, 保证 a/b/c/d 左对齐且左侧无任何文字 ----------
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_W = fig.get_figwidth() * fig.dpi
_lefts = []
for _ax in fig.axes:
    _arts = (list(_ax.texts) + [_ax.title, _ax.xaxis.label, _ax.yaxis.label]
             + list(_ax.get_xticklabels()) + list(_ax.get_yticklabels()))
    for _t in _arts:
        if _t.get_text().strip():
            _lefts.append(_t.get_window_extent(renderer=_r).x0)
for _t in fig.texts:
    if _t.get_text().strip():
        _lefts.append(_t.get_window_extent(renderer=_r).x0)
for _lab, _ax in [("a", axa), ("b", axb), ("c", axc), ("d", AXP)]:
    fig.text(min(_lefts) / _W - 0.012, _ax.get_position().y1 + 0.008, _lab,
             fontsize=17, fontweight="bold")

out = os.path.join(HERE, "fig7_extrap_maize.svg")
fig.savefig(out, format="svg", bbox_inches="tight")
print("saved ->", out)
