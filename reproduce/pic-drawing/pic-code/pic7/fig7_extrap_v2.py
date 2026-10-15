# -*- coding: utf-8 -*-
"""fig7_extrap_v2.py — 图七 外推 (forecasting) 主图 v2 (2026-09-20: 并入新跑的作者 CV3 数值), 四个数据集共用一份脚本 (照 fig7_extrap_maize.py 的版式)
用法: python fig7_extrap.py --dataset maize|arab|rice|uav
数据 (同目录, 由 make_fig7_data.py 生成):
  fig7_<ds>_b.csv  逐 DAS 的 mean/sd PCC/MSE (rec_tk; 有作者冻结值的数据集里 method 会有两行)
  fig7_<ds>_c.csv  逐 trait 的 MSE (rec_tk)
  fig7_<ds>_d.csv  三条件总体 mean/sd (rec_t1 / rec_tk / iter)
输出: fig7_extrap_<ds>.svg
"""
import os, argparse
import numpy as np, pandas as pd, matplotlib
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
C_K, C_D = "#D62728", "#1F77B4"
LAB_K, LAB_D = "Kode", "dynamicGP-MegaLMM+TP1"
# 每个数据集: n_test(尾部留出点数) + 全部 DAS (从数据里读) + 面板 d 的横轴范围 (None=自适应)
SPECS = {
    "maize": dict(n_test=5, name="Maize MAGIC"),
    "arab":  dict(n_test=2, name="Arabidopsis"),
    "rice":  dict(n_test=2, name="Rice (control)"),
    "uav":   dict(n_test=2, name="UAV RGB (drought)"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="maize", choices=list(SPECS))
a = ap.parse_args()
DS = a.dataset
spec = SPECS[DS]
B = pd.read_csv(f"{HERE}/fig7_{DS}_b_v2.csv")
C = pd.read_csv(f"{HERE}/fig7_{DS}_c_v2.csv")
Dd = pd.read_csv(f"{HERE}/fig7_{DS}_d_v2.csv")
DAS = sorted(B[B.method == LAB_K]["DAS"].astype(int).unique())
T = len(DAS); NT = int(spec["n_test"]); FC = DAS[-NT:]; TK = T - NT
ANCHOR = DAS[TK - 1]
HAS_AU = LAB_D in set(B.method)
print(f"[{DS}] T={T} DAS={DAS} | anchor t{TK}=DAS{ANCHOR} | horizon={FC} | 作者方法: {HAS_AU}")


def style(ax, title, ylab=None, xlab=None):
    if ylab: ax.set_ylabel(ylab, fontweight="bold", fontsize=10)
    if xlab: ax.set_xlabel(xlab, fontweight="bold", fontsize=10)
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left", pad=6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=3)
    for lab in ax.get_xticklabels() + ax.get_yticklabels(): lab.set_fontweight("bold")


fig = plt.figure(figsize=(7.0, 11.0))
gs = fig.add_gridspec(4, 2, height_ratios=[0.82, 1.0, 1.15, 1.0], hspace=0.66, wspace=0.28,
                      left=0.135, right=0.985, top=0.93, bottom=0.065)

# ---------- a) 协议示意 (按数据集自适应: T 个时间点, 前 TK 训练, 后 NT 预测) ----------
axa = fig.add_subplot(gs[0, :]); axa.set_xlim(-0.6, T + 0.4); axa.set_ylim(-1.5, 2.5); axa.axis("off")
wtrain = TK - 0.5
axa.add_patch(Rectangle((-0.5, 0.05), wtrain + 0.5, 0.9, facecolor="#EAF1FA", edgecolor="none", zorder=0))
axa.add_patch(Rectangle((wtrain, 0.05), T - wtrain + 0.5, 0.9, facecolor="#FBE9E7", edgecolor="none", zorder=0))
axa.text(wtrain / 2, 1.35, f"training window  t1-t{TK}  (DAS{DAS[0]}-{ANCHOR})", ha="center",
         fontsize=8.4, color="#2C5D9E", fontweight="bold")
axa.text(wtrain + (T - wtrain) / 2, 1.35, f"forecast horizon  t{TK+1}-t{T}  (DAS{FC[0]}-{FC[-1]})",
         ha="center", fontsize=8.4, color="#B23A2E", fontweight="bold")
for i in range(T):
    axa.plot([i + 0.5], [0.5], marker="o", ms=3.6 if i < TK else 3.0,
             mfc="#3C6DB0" if i < TK else "white", mec="#3C6DB0", mew=0.9, zorder=3)
axa.plot([wtrain], [0.5], marker="o", ms=7.0, mfc=C_K, mec="white", mew=0.8, zorder=5)
axa.add_patch(FancyArrowPatch((wtrain, -0.35), (wtrain, 0.28), arrowstyle="-|>", mutation_scale=7,
                             lw=1.0, color=C_K, zorder=5))
axa.text(wtrain, -0.62, f"true t{TK}\n(forecast anchor)", ha="center", va="top", fontsize=7.6,
         color=C_K, fontweight="bold")
axa.text(wtrain + (T - wtrain) / 2, -0.62, f"{NT} unseen timepoint{'s' if NT > 1 else ''}\n"
         + ("(all methods)" if HAS_AU else "(Kode)"), ha="center", va="top", fontsize=7.6, color="#B23A2E")
axa.text(-0.4, -1.25, "other settings:  t1-only rollout (autonomous)   ·   iterative (observation-assisted)",
         ha="left", va="top", fontsize=7.4, color="0.35", style="italic")

# ---------- b) PCC (线 + ribbon) ----------
axb = fig.add_subplot(gs[1, :])
series = [(LAB_D, C_D, "--")] if HAS_AU else []
series += [(LAB_K, C_K, "-")]
for who, col, ls in series:
    g = B[B.method == who].sort_values("DAS"); g = g[g.DAS.isin(FC)]
    axb.fill_between(g.DAS, g.pcc_mean - g.pcc_sd, g.pcc_mean + g.pcc_sd, color=col, alpha=0.20, lw=0, zorder=2)
    axb.plot(g.DAS, g.pcc_mean, ls=ls, lw=1.9 if who == LAB_K else 1.6, marker="o", ms=2.6,
             mec=col, mfc=col, mew=0.25, label=who, zorder=3)
style(axb, "Matched CV3 forecasting — PCC", "Prediction accuracy (r)", "DAS")
axb.set_xticks(FC)
_vals = [v for who, _c, _l in series for v in B[B.method == who].pcc_mean if np.isfinite(v)]
axb.set_ylim(max(0.0, min(_vals) - 0.08), min(1.02, max(_vals) + 0.08))
axb.legend(frameon=False, loc="lower left")

# ---------- c) MSE (log10 分组箱线) ----------
axc = fig.add_subplot(gs[2, :])
offs = [0.0] if not HAS_AU else [-0.20, +0.20]
for (who, col, ls), off in zip(series, offs if HAS_AU else [0.0]):
    for i, t in enumerate(FC):
        v = C[(C.method == who) & (C.time == t)]["mse"].values.astype(float)
        v = np.log10(v[np.isfinite(v) & (v > 0)])
        if v.size == 0: continue
        bp = axc.boxplot([v], positions=[i + off], widths=0.15 if HAS_AU else 0.34, patch_artist=True,
                         manage_ticks=False, showfliers=True,
                         flierprops=dict(marker="o", markersize=0.8, markerfacecolor="0.35", markeredgecolor="none", alpha=0.1),
                         whiskerprops=dict(linewidth=0.35, color="0.25", linestyle=ls),
                         capprops=dict(linewidth=0.35, color="0.25", linestyle=ls),
                         boxprops=dict(linewidth=0.35, linestyle=ls),
                         medianprops=dict(linewidth=0.6, color="0.1", linestyle=ls))
        bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_edgecolor(col); bp["boxes"][0].set_alpha(0.80)
style(axc, "Matched CV3 forecasting — MSE", r"$\log_{10}$(MSE)", "DAS")
axc.set_xticks(range(len(FC))); axc.set_xticklabels([str(t) for t in FC])
lg = np.log10(C.mse[C.mse > 0])
axc.set_ylim(np.floor(lg.quantile(0.01)) - 0.2, np.ceil(lg.quantile(0.99)) + 0.2)
axc.legend(handles=[Patch(facecolor=C_K, edgecolor=C_K, alpha=0.80, label=LAB_K)]
           + ([Patch(facecolor=C_D, edgecolor=C_D, alpha=0.80, linestyle="--", label=LAB_D)] if HAS_AU else []),
           frameon=False, loc="lower left", fontsize=8.0, handlelength=1.5, borderpad=0.2, labelspacing=0.3)

# ---------- d) 三条件 paired dot ----------
AXP = fig.add_subplot(gs[3, 0]); AXM = fig.add_subplot(gs[3, 1])
SC = [("rec_t1", "t1-only"), ("rec_tk", f"t{TK}-anchored"), ("iter", "iterative\n(assisted)")]
ypos = np.arange(len(SC))[::-1]


def paired(ax, metric, xlab, xlim):
    for y, (s, _sl) in zip(ypos, SC):
        for who, col in series_labels:
            r = Dd[(Dd.method == who) & (Dd.scenario == s)]
            if len(r) == 0: continue
            r = r.iloc[0]
            ax.errorbar([r[f"{metric}_mean"]], [y], xerr=[r[f"{metric}_sd"]], fmt="o", ms=5.0, color=col,
                        mec="white", mew=0.6, capsize=2.0, elinewidth=0.8, zorder=3)
            ax.text(r[f"{metric}_mean"], y + 0.19, "%.3f" % r[f"{metric}_mean"], ha="center", fontsize=7.0,
                    color=col, fontweight="bold")
    ax.set_yticks(ypos); ax.set_yticklabels([s for _k, s in SC], fontsize=8.6)
    ax.set_ylim(-0.55, len(SC) - 0.45); ax.set_xlim(*xlim)
    style(ax, xlab, None, None)


series_labels = [(who, col) for who, col, _ls in series]
_all = [v for m in ("pcc", "mse") for who, _c in series_labels for s, _sl in SC
        for v in [Dd[(Dd.method == who) & (Dd.scenario == s)][f"{m}_mean"]] if len(v)]
px = [float(v.iloc[0]) for v in _all[:len(_all) // 2]] or [0.3]
mx = [float(v.iloc[0]) for v in _all[len(_all) // 2:]] or [0.01]
paired(AXP, "pcc", f"mean PCC ({FC[0]}-{FC[-1]})", (max(0.0, min(px) - 0.06), min(1.02, max(px) + 0.06)))
paired(AXM, "mse", f"mean MSE ({FC[0]}-{FC[-1]})", (max(0.0, min(mx) * 0.7), max(mx) * 1.3))
AXP.set_title("Overall — mean PCC", fontsize=10, fontweight="bold", loc="left", pad=5)
AXM.set_title("Overall — mean MSE", fontsize=10, fontweight="bold", loc="left", pad=5)
AXM.set_yticklabels([])
AXP.legend(handles=[Line2D([], [], marker="o", ls="", ms=5.0, color=c, mec="white", label=w)
                    for w, c in series_labels], frameon=False, loc="upper right", fontsize=7.6)

# ---------- 统一面板字母 ----------
fig.canvas.draw()
_r = fig.canvas.get_renderer(); _W = fig.get_figwidth() * fig.dpi; _lefts = []
for _ax in fig.axes:
    _arts = (list(_ax.texts) + [_ax.title, _ax.xaxis.label, _ax.yaxis.label]
             + list(_ax.get_xticklabels()) + list(_ax.get_yticklabels()))
    for _t in _arts:
        if _t.get_text().strip(): _lefts.append(_t.get_window_extent(renderer=_r).x0)
for _t in fig.texts:
    if _t.get_text().strip(): _lefts.append(_t.get_window_extent(renderer=_r).x0)
for _lab, _ax in [("a", axa), ("b", axb), ("c", axc), ("d", AXP)]:
    fig.text(min(_lefts) / _W - 0.012, _ax.get_position().y1 + 0.008, _lab, fontsize=17, fontweight="bold")

out = os.path.join(HERE, f"fig7_extrap_{DS}_v2.svg")
fig.savefig(out, format="svg", bbox_inches="tight")
print("saved ->", out)