# -*- coding: utf-8 -*-
"""fig_rank.py — rank 分析 (玉米; abc 三面板 5 条线)

数据 (同目录): fig_rank_data.csv   长表: panel, dataset, series, x, mean, sd
输出 (同目录): fig_rank.svg

  a 轨迹预测 PCC vs rank          full / tp1 / grm / shared
  b 算子 A 的 SNP 可预测性 vs rank  同上
  c 累积谱能量 vs 模态数            full / tp1 / grm / shared —— 仅玉米

series 的线型/颜色在 SERIES_STYLE 里改; 数据里没有的 series 自动跳过。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "fig_rank_data.csv")
DS   = "maize"          # 本图只用玉米

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10.5, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "savefig.dpi": 300, "figure.dpi": 110, "svg.fonttype": "none",
})

# 画图顺序 + 样式（要改线/加线就改这里）
SERIES_STYLE = {
    "full":      dict(color="#d62728", marker="o", ls="-",  lw=2.2, ms=6.0,
                      label="GRM+TP1-conditioned Kode"),
    "shared":    dict(color="#b8b8b8", marker="s", ls="-",  lw=1.8, ms=5.5,
                      label="Shared dynamics"),
    "tp1":       dict(color="#1f78d1", marker="^", ls="-",  lw=1.8, ms=5.5,
                      label="TP1-conditioned"),
    "grm":       dict(color="#2ca02c", marker="v", ls="-",  lw=1.8, ms=5.5,
                      label="GRM-conditioned"),
}
ORDER = ["full", "tp1", "grm", "shared"]
# 某些 panel 不画某些 series: 全0(shared) 的算子无方差, 求 SNP 可预测性无定义
EXCLUDE = {"b": {"shared"}}

def get(d, panel, s):
    sub = d[(d.panel == panel) & (d.dataset == DS) & (d.series == s)].sort_values("x")
    return sub["x"].values, sub["mean"].values, sub["sd"].values

def draw(ax, d, panel, with_band=True):
    """按 ORDER 画所有存在于数据里的 series, 返回图例句柄"""
    handles = []
    excl = EXCLUDE.get(panel, set())
    for s in ORDER:
        if s in excl:
            continue
        x, y, sd = get(d, panel, s)
        if len(x) == 0 or not np.isfinite(y).any():
            continue
        st = SERIES_STYLE[s]
        if with_band and np.isfinite(sd).any():
            ax.fill_between(x, y - sd, y + sd, color=st["color"], alpha=0.13, lw=0, zorder=2)
        ln, = ax.plot(x, y, marker=st["marker"], ms=st["ms"], lw=st["lw"],
                      ls=st["ls"], color=st["color"], zorder=3)
        handles.append(ln)
    return handles

def main():
    d = pd.read_csv(CSV)
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = GridSpec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0],
                  wspace=0.20, hspace=0.38)
    ax1 = fig.add_subplot(gs[:, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[1, 1])

    # ---- a ----
    h = draw(ax1, d, "a")
    ax1.axvline(2, ls="--", lw=1.4, color="gray", alpha=0.85, zorder=1)
    ax1.text(2.8, 0.592, r"$r=2$" + "\n(selected)", fontsize=10.5, va="top")
    ax1.set_title("Trajectory prediction saturates at low rank", loc="left",
                  fontsize=14, fontweight="bold", pad=10)
    ax1.set_xlabel(r"Operator rank ($r$)", fontsize=12); ax1.set_ylabel("Recursive PCC", fontsize=12)
    ax1.set_xlim(0.5, 53); ax1.set_ylim(0.30, 0.62)
    xr = get(d, "a", "full")[0]
    ax1.set_xticks(xr)
    ax1.legend(handles=h, labels=[SERIES_STYLE[s]["label"] for s in ORDER
               if len(get(d, "a", s)[0])], frameon=False, fontsize=10.5, loc="lower right")

    # ---- b ----
    h = draw(ax2, d, "b")
    ax2.set_title("Operator predictability vs. rank", loc="left",
                  fontsize=14, fontweight="bold", pad=10)
    ax2.set_ylabel(r"SNP predictability of $A$ (r)", fontsize=12)
    ax2.set_xlim(0.5, 53); ax2.set_ylim(0.3, 1.0)
    ax2.set_xticks(xr)
    ax2.legend(handles=h, labels=[SERIES_STYLE[s]["label"] for s in ORDER
               if s not in EXCLUDE.get("b", set()) and len(get(d, "b", s)[0])],
               frameon=False, fontsize=10.5, loc="center right")

    # ---- c (仅玉米) ----
    h = draw(ax3, d, "c", with_band=False)
    # ---- rank=2 时 full 算子所占的谱能量份额 (不再标 90% 线: 想说明的是 r=2 已占很大份额) ----
    _xf, _yf, _ = get(d, "c", "full")
    _k2 = int(np.where(_xf == 2)[0][0])
    _y2 = float(_yf[_k2])
    ax3.hlines(_y2, 0.5, 2, ls=":", lw=1.2, color="0.55", zorder=1)
    ax3.plot([2], [_y2], marker="o", ms=7, mfc="white",
             mec=SERIES_STYLE["full"]["color"], mew=2.0, zorder=5)
    # 2026-09-22: 原来放在 x=2.5 会压在 k=2~4 的曲线交叠区上, 右移到曲线下方的空白楔形区
    ax3.text(4.6, _y2, r"$r=2$: %.1f%%" % (100 * _y2), fontsize=10.5, va="center", color="0.25")
    ax3.set_title("Operator spectrum is low-dimensional", loc="left",
                  fontsize=14, fontweight="bold", pad=10)
    ax3.set_xlabel(r"Number of singular modes ($k$)", fontsize=12)
    ax3.set_ylabel("Cumulative spectral energy", fontsize=12)
    # y 轴自适应: 按 x 显示范围内的所有点定下限, 避免低 k 的点被裁掉
    _vals = []
    for _s in ORDER:
        _xs, _ys, _sd = get(d, "c", _s)
        _vals += [float(_v) for _xx, _v in zip(_xs, _ys) if _xx <= 12 and np.isfinite(_v)]
    ax3.set_xlim(0.5, 12); ax3.set_ylim(min(_vals) - 0.05, 1.02)
    ax3.set_xticks([1,2,3,4,5,8,11])
    ax3.legend(handles=h, labels=[SERIES_STYLE[s]["label"] for s in ORDER
               if len(get(d, "c", s)[0])], frameon=False, fontsize=10.5, loc="lower right")

    for ax, lab in [(ax1, "a"), (ax2, "b"), (ax3, "c")]:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontweight("bold")
        ax.text(-0.13, 1.03, lab, transform=ax.transAxes, fontsize=20, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(HERE, "fig_rank.svg")
    fig.savefig(out, bbox_inches="tight")
    print("saved ->", out)
    plt.close(fig)

if __name__ == "__main__":
    print("data ->", CSV, "| dataset =", DS)
    main()
