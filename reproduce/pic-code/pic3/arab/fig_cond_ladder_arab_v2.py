# -*- coding: utf-8 -*-
"""fig_cond_ladder_maize.py — Learned developmental dynamics go beyond early-state persistence
(Arabidopsis diversity panel, 10 fold-sets, rank=2; 数据全部来自本目录 CSV)

数据 (同目录):
  knode_cond_ladder_maize_agg.csv           条件级 PCC/MSE/NRMSE/R2
  knode_cond_ladder_maize_f0..f9.csv        逐折条件级 (算 MSE SD)
  knode_cond_ladder_maize_trait_f0..f9.csv  逐 trait x timepoint x fold
输出:
  fig_cond_ladder_maize.png / .svg / .pdf
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10.5,
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "savefig.dpi": 300, "figure.dpi": 110, "svg.fonttype": "none",
})

# =========================
# 1. Overall results  (真实数据)
# =========================
KEYS   = ["persistence", "shared", "grm", "tp1", "full"]
models = ["TP1\npersistence", "Shared\ndynamics", "GRM-\nconditioned", "TP1-\nconditioned", "GRM+TP1-\nconditioned Kode"]

DATA = os.path.join(HERE, "fig_cond_ladder_arab_data.csv")
assert os.path.isfile(DATA), "缺少汇总数据 %s -- 请先运行 make_fig_cond_ladder_data.py" % DATA
raw = pd.read_csv(DATA)

ov = raw[raw["block"] == "overall"].set_index("cond")
pcc_mean = np.array([float(ov.loc[k, "pcc"])    for k in KEYS])
pcc_sd   = np.array([float(ov.loc[k, "pcc_sd"]) for k in KEYS])
mse_mean = np.array([float(ov.loc[k, "mse"])    for k in KEYS])
mse_sd   = np.array([float(ov.loc[k, "mse_sd"]) for k in KEYS])
# 2026-09-21: MSE switched to log10 (unified with fig2/5/6/8)
MSE_LO, MSE_HI = -2.40, -0.72
MSE_TICKS = [-2.0, -1.5, -1.0]
mse_l  = np.log10(mse_mean)
mse_lo = np.log10(np.maximum(mse_mean - mse_sd, 1e-12))
mse_hi = np.log10(mse_mean + mse_sd)
mse_err = [mse_l - mse_lo, mse_hi - mse_l]

# 2026-09-22: 低饱和柔和配色 (Paired-light 系), 红/紫已对调: GRM-conditioned=紫, Full Kode=红
colors = {
    "persistence": "#D9D9D9",   # 灰
    "shared":      "#A6CEE3",   # 淡蓝
    "grm":         "#CAB2D6",   # 淡紫
    "tp1":         "#B2DF8A",   # 淡绿
    "full":        "#FB9A99",   # 淡红  (Full Kode)
}
bar_colors = [colors[k] for k in KEYS]

# =========================
# 2. Trait-level results (真实数据)
# =========================
tr = raw[raw["block"] == "trait"]
trait_pcc = tr.pivot_table(index="trait", columns="cond", values="pcc")
trait_mse = tr.pivot_table(index="trait", columns="cond", values="mse")

trait_pcc_persist = trait_pcc["persistence"].values
trait_pcc_full    = trait_pcc["full"].values
trait_mse_persist = trait_mse["persistence"].values
trait_mse_full    = trait_mse["full"].values
n_traits = len(trait_pcc_persist)

mse_improve_ratio = trait_mse_persist / trait_mse_full      # >1 表示 Full 更好

# =========================
# 3. Figure layout
# =========================
fig = plt.figure(figsize=(14.5, 8.5))
gs = GridSpec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0],
              wspace=0.25, hspace=0.35)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

# =========================
# Panel A: Overall PCC
# =========================
x = np.arange(len(models))
axA.bar(x, pcc_mean, yerr=pcc_sd, capsize=4, width=0.68,
        color=bar_colors, edgecolor='none')
axA.set_xticks(x); axA.set_xticklabels(models, fontsize=10.5, fontweight="bold")
axA.set_ylabel("Prediction accuracy (PCC)", fontsize=12)
axA.set_title("Overall prediction accuracy", loc='left', fontsize=14, weight='bold')
for i, v in enumerate(pcc_mean):
    axA.text(i, v + pcc_sd[i] + 0.012, f"{v:.3f}", ha='center', va='bottom',
             fontsize=10.5, fontweight='bold')

# persistence vs full 括号 (自适应高度)
top_a = float(np.max(pcc_mean + pcc_sd))
y_br  = top_a + 0.045
axA.plot([0, 0, 4, 4], [y_br-0.02, y_br, y_br, y_br-0.02], color="#555555", lw=1.2)
axA.text(2, y_br+0.008, r"$\Delta$PCC = %+.3f" % (pcc_mean[4]-pcc_mean[0]),
         ha='center', va='bottom', fontsize=10.5, fontweight='bold')
axA.set_ylim(0, y_br + 0.075)
axA.text(-0.15, 1.07, "a", transform=axA.transAxes, fontsize=20, weight='bold')

# =========================
# Panel B: Overall MSE (log scale)
# =========================
axB.bar(x, mse_l, yerr=mse_err, capsize=4, width=0.68,
        color=bar_colors, edgecolor='none')
axB.set_xticks(x); axB.set_xticklabels(models, fontsize=10.5, fontweight="bold")
axB.set_ylabel(r'$\log_{10}$(MSE)', fontsize=12)
axB.set_title("Recovery of phenotype magnitude", loc='left', fontsize=14, weight='bold')
for i, v in enumerate(mse_l):
    axB.text(i, v + 0.06, f"{v:.2f}", ha='center', va='bottom',
             fontsize=10.5, fontweight='bold')
# log 轴: 只保留 10^-2 / 10^-1 两条主线, 并关掉会显得不等距的次刻度
axB.set_ylim(MSE_LO, MSE_HI)          # 上限 = 10^-1 再高 20%
axB.set_yticks(MSE_TICKS)
axB.text(-0.15, 1.07, "b", transform=axB.transAxes, fontsize=20, weight='bold')

# =========================
# Panel C: Full vs Persistence by trait
# =========================
sc = axC.scatter(trait_pcc_persist, trait_pcc_full,
                 c=mse_improve_ratio, cmap="viridis",
                 norm=LogNorm(vmin=1.0, vmax=float(mse_improve_ratio.max())),
                 s=60, alpha=0.9, edgecolor='white', linewidth=0.5)
minv = min(trait_pcc_persist.min(), trait_pcc_full.min()) - 0.03
maxv = max(trait_pcc_persist.max(), trait_pcc_full.max()) + 0.03
axC.plot([minv, maxv], [minv, maxv], linestyle='--', color='gray', lw=1.2)
axC.set_xlim(minv, maxv); axC.set_ylim(minv, maxv)
axC.set_xlabel("Trait-level PCC (TP1 persistence)", fontsize=12)
axC.set_ylabel("Trait-level PCC (GRM+TP1-\nconditioned Kode)", fontsize=12)
axC.set_title("Trait-level improvement beyond persistence", loc='left', fontsize=14, weight='bold')
n_above = int(np.sum(trait_pcc_full > trait_pcc_persist))
axC.text(0.04, 0.92, f"{n_above}/{n_traits} traits above identity line",
         transform=axC.transAxes, fontsize=10.5, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor='none'))
cbar = fig.colorbar(sc, ax=axC, fraction=0.046, pad=0.04)
cbar.set_label("MSE improvement\n(fold-change,\nTP1 persistence /\nGRM+TP1-conditioned Kode)", fontsize=9.5, fontweight="bold")
cbar.set_ticks([1, 2, 5, 10, 20, 50])
cbar.set_ticklabels(["1×", "2×", "5×", "10×", "20×", "50×"])
for _t in cbar.ax.get_yticklabels():
    _t.set_fontsize(9); _t.set_fontweight('bold')
cbar.ax.set_title(">1× = Kode\nbetter", fontsize=9, fontweight='bold', pad=6)
axC.text(-0.15, 1.07, "c", transform=axC.transAxes, fontsize=20, weight='bold')

# =========================
# Panel D: logic diagram
# =========================
axD.set_axis_off(); axD.set_xlim(-0.2, 9.6); axD.set_ylim(0.4, 9.2)
axD.set_title("From persistence to individualized dynamics", loc='left', fontsize=14, weight='bold')

def add_box(ax, xy, w, h, text, facecolor, fontsize=11):
    box = FancyBboxPatch(xy, w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.12",
                         linewidth=1.0, edgecolor='none', facecolor=facecolor)
    ax.add_patch(box)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold')

def add_arrow(ax, start, end, color="#666666", lw=1.5):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                 mutation_scale=12, linewidth=lw, color=color))

# 框放大, 且只保留框内文字 (删掉所有旁注)
add_box(axD, (0.35, 7.00), 2.9, 1.65, "TP1\npersistence", colors["persistence"], fontsize=12)
add_box(axD, (3.85, 7.00), 2.9, 1.65, "Shared\ndynamics", colors["shared"], fontsize=12)
add_box(axD, (1.35, 4.00), 2.9, 1.65, "GRM-\nconditioned", colors["grm"], fontsize=12)
add_box(axD, (5.85, 4.00), 2.9, 1.65, "TP1-\nconditioned", colors["tp1"], fontsize=12)
add_box(axD, (3.60, 0.90), 2.9, 1.75, "GRM+TP1-\nconditioned Kode", colors["full"], fontsize=12)

add_arrow(axD, (3.25, 7.83), (3.85, 7.83))
add_arrow(axD, (4.30, 7.00), (3.30, 5.65), color=colors["grm"])
add_arrow(axD, (6.30, 7.00), (6.90, 5.65), color=colors["tp1"])
add_arrow(axD, (2.30, 4.00), (4.30, 2.65), color=colors["grm"])
add_arrow(axD, (7.30, 4.00), (5.80, 2.65), color=colors["tp1"])
axD.text(-0.10, 1.07, "d", transform=axD.transAxes, fontsize=20, weight='bold')

# =========================
# Styling
# =========================
for ax in [axA, axB, axC]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight('bold')

# 不加整图标题: 只保留各子图标题
fig.subplots_adjust(left=0.055, right=0.985, top=0.955, bottom=0.075, wspace=0.24, hspace=0.34)
for ext in ("svg",):   # 只输出 SVG
    out = os.path.join(HERE, f"fig_cond_ladder_arab_v2.{ext}")
    fig.savefig(out, bbox_inches="tight")
    print("saved ->", out)
plt.close(fig)

print("PCC :", dict(zip(models, np.round(pcc_mean, 4))))
print("PCCsd:", dict(zip(models, np.round(pcc_sd, 4))))
print("MSE :", dict(zip(models, np.round(mse_mean, 5))))
print("MSEsd:", dict(zip(models, np.round(mse_sd, 5))))
print("dPCC persistence->full = %+.4f" % (pcc_mean[4]-pcc_mean[0]))
print("traits above identity  = %d/%d" % (n_above, n_traits))
print("MSE improvement ratio  = %.2f ~ %.2f" % (mse_improve_ratio.min(), mse_improve_ratio.max()))
