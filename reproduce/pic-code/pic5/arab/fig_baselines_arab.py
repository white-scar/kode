# -*- coding: utf-8 -*-
"""fig_baselines_arab_v2.py — Kode vs 动态方法基线（拟南芥）

数据 (同目录):
  fig_baselines_data.csv       逐时间点 PCC/MSE 均值+SD (panel a)
  fig_baselines_mse_long.csv   逐 trait x fold 的 MSE   (panel b 箱线图)
输出: fig_baselines_arab.svg

竖排 2 行: (a) 逐时间点 PCC | (b) log10(MSE) 分组箱线图
x 轴为真实 DAS (按真实天数排布, 非等距)
曲线/箱: Kode rec / Kode iter (红, 实/虚) + 4 条动态基线 (避开蓝绿)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "fig_baselines_data.csv")
LONG = os.path.join(HERE, "fig_baselines_mse_long.csv")

# x 轴直接使用真实 DAS(单位: 天), 不再做 t1..t25 索引化;
# 采样间隔本就 1/3 天不等, 真实时间轴会把缺口如实呈现

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10.5, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "savefig.dpi": 300, "figure.dpi": 110, "svg.fonttype": "none",
})

RED = "#d62728"
# 基线配色刻意避开蓝/绿 (蓝绿留给作者方法)
STYLE = {
    "Kode_rec":        dict(color=RED,       ls="-",  lw=2.4, ms=4.6, label="Kode (recursive)"),
    "Kode_iter":       dict(color=RED,       ls=(0, (5.0, 2.2)), lw=2.4, ms=4.6, label="Kode (iterative)"),
    # 基线: 橙 / 玫红 / 中灰 / 近黑 —— 彼此差异大, 且都避开蓝绿与红(留给 Kode)
    "lstm":            dict(color="#E69F00", ls="-",  lw=1.7, ms=3.4, label="LSTM"),
    "ar_transformer":  dict(color="#CC79A7", ls="-",  lw=1.7, ms=3.4, label="AR-Transformer"),
    "ode_rnn":         dict(color="#8C8C8C", ls="-",  lw=1.7, ms=3.4, label="ODE-RNN"),
    # 注: 该实现是 GRU-ODE 连续演化 + GRUCell jump, 不含原版 GRU-ODE-Bayes 的贝叶斯观测更新机制,
    #     故论文中统一称 "GRU-ODE-Style" (数据里的 method 名也已同步改成 GRU-ODE-Style)
    "GRU-ODE-Style":  dict(color="#2B2B2B", ls="-",  lw=1.7, ms=3.4, label="GRU-ODE-Style"),
    "ststp":           dict(color="#B0AFAF", ls="-",  lw=1.7, ms=3.4, label="ST-STP"),
}
ORDER = ["Kode_rec", "Kode_iter", "lstm", "ar_transformer", "ode_rnn", "GRU-ODE-Style"]
DODGE = {m: off for m, off in zip(ORDER, np.linspace(-0.30, 0.30, len(ORDER)))}
W = 0.088

def draw_box(ax, long, ds, ts):
    data, pos, cols, lss = [], [], [], []
    for tp in ts:
        xpos = float(tp)              # 真实 DAS
        for m in ORDER:
            v = long[(long.dataset == ds) & (long.method == m) & (long.time == tp)]["mse"].values
            v = v[np.isfinite(v) & (v > 0)]
            if v.size == 0:
                v = np.array([np.nan])
            data.append(np.log10(v + 1e-12))
            pos.append(xpos + DODGE[m]); cols.append(STYLE[m]["color"]); lss.append(STYLE[m]["ls"])
    bp = ax.boxplot(data, positions=pos, widths=W, patch_artist=True,
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
    d = pd.read_csv(CSV); long = pd.read_csv(LONG)
    ds = "arab"
    sub = d[d.dataset == ds].copy()
    sub["t"] = sub["time"].astype(float)      # 真实 DAS
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.4))

    # ---- a: PCC ----
    ax = axes[0]
    for m in ORDER:
        s = sub[sub.method == m].sort_values("t")
        if not len(s): continue
        st = STYLE[m]
        x, y, sd = s["t"].values, s["pcc"].values, s["pcc_sd"].values
        ax.fill_between(x, y - sd, y + sd, color=st["color"],
                        alpha=(0.18 if m.startswith("Kode") else 0.10), lw=0)
        ax.plot(x, y, color=st["color"], ls=st["ls"], lw=st["lw"], marker="o",
                ms=st["ms"], label=st["label"], zorder=(5 if m.startswith("Kode") else 3))
    ax.set_title("Arabidopsis — PCC", fontsize=14, fontweight="bold", loc="left", pad=6)
    ax.text(-0.14, 1.04, "a", transform=ax.transAxes, fontsize=20, fontweight="bold")
    ax.set_ylabel("Prediction accuracy (PCC)", fontsize=12)
    ax.set_ylim(0.15, 0.95)

    # ---- b: MSE 箱线图 ----
    ax = axes[1]
    raw_t = sorted(long[long.dataset == ds]["time"].unique())
    draw_box(ax, long, ds, raw_t)
    ax.set_title("Arabidopsis — MSE", fontsize=14, fontweight="bold", loc="left", pad=6)
    ax.text(-0.14, 1.04, "b", transform=ax.transAxes, fontsize=20, fontweight="bold")
    ax.set_ylabel(r"$\log_{10}$(MSE)", fontsize=12)
    ax.set_ylim(-4.0, 0.2); ax.set_yticks(list(range(-4, 1)))

    for ax in axes:
        ax.set_xlabel("DAS", fontsize=12)
        ax.set_xticks(raw_t)
        ax.set_xticklabels([str(int(v)) for v in raw_t], fontsize=9,
                           rotation=45, ha="right")
        ax.set_xlim(raw_t[0] - 0.7, raw_t[-1] + 0.7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=3, labelsize=9)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontweight("bold")

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                     fontsize=10.5, bbox_to_anchor=(0.5, -0.02), handlelength=2.0)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    fig.tight_layout(rect=[0, 0.045, 1, 1])
    # v2: 用当前 fig_baselines_data.csv 重画 (2026-09-22)
    out = os.path.join(HERE, "fig_baselines_arab.svg")
    fig.savefig(out, bbox_inches="tight")
    print("saved ->", out)
    plt.close(fig)

if __name__ == "__main__":
    print("data ->", CSV, "| long ->", LONG)
    main()
