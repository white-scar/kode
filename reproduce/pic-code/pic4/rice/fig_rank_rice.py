# -*- coding: utf-8 -*-
"""fig_rank_rice.py — 水稻 Fig.4 (rank 分析): **一个脚本出 a / b / c 三个面板**。

数据全部自动读取 (与 fig_rank_arab.py 同一模板):
  panel a : outputs/reproduce/rank_sweep_rice_dg10/agg.csv   -> pcc_mean / pcc_std   (轨迹预测 PCC vs rank)
  panel b : 同上                               -> ext_A_mean / ext_A_std (算子 A 的 SNP 可预测性 vs rank)
  panel c : 同目录 fig_rank_rice_energy.csv (cond, k, energy);
            没有则从 outputs/reproduce/rank_sweep_rice_dg10/cache/f*_{cond}_r20_i*_ep1000.npz 现算累积谱能量
            (与玉米 panel c 同一口径, 最大 rank 20=P)

输出 (同目录):
  fig_rank_rice.svg        只出 SVG
  fig_rank_rice_data.csv   画图长表 (panel, dataset, series, x, mean, sd)  —— 便于复核

线名与图三 (cond ladder) 统一:
  full -> GRM+TP1-conditioned Kode | tp1 -> TP1-conditioned | grm -> GRM-conditioned | const -> Shared dynamics
注意: rice 的 full 低于 tp1only/const, 即 GRM 在该数据集上是负贡献 (与 cond ladder 一致)。
"""
import os, glob, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
RUN  = os.path.join(BASE, "outputs", "reproduce", "rank_sweep_rice_dg10")
AGG  = os.path.join(RUN, "agg.csv")
CACHE= os.path.join(RUN, "cache")
OUT  = os.path.join(HERE, "fig_rank_rice.svg")
DATA = os.path.join(HERE, "fig_rank_rice_data.csv")
ENERG= os.path.join(HERE, "fig_rank_rice_energy.csv")   # panel c 现成谱能量 (没有才从 cache 现算)
DS   = "rice"
MAXR = 20                                  # 谱能量用最大 rank (=P=20) 的 cache
KS   = [1, 2, 3, 4, 6, 8, 10, 16, 20]   # 与 rank 网格一致
COND2SERIES = {"full": "full", "tp1only": "tp1", "grmonly": "grm", "const": "shared"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10.5, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "savefig.dpi": 300, "figure.dpi": 110, "svg.fonttype": "none",
})

# 线名/线型 (与图三统一; 要改线就改这里)
SERIES_STYLE = {
    "full":   dict(color="#d62728", marker="o", ls="-", lw=2.2, ms=6.0, label="GRM+TP1-conditioned Kode"),
    "tp1":    dict(color="#1f78d1", marker="^", ls="-", lw=1.8, ms=5.5, label="TP1-conditioned"),
    "grm":    dict(color="#2ca02c", marker="v", ls="-", lw=1.8, ms=5.5, label="GRM-conditioned"),
    "shared": dict(color="#b8b8b8", marker="s", ls="-", lw=1.8, ms=5.5, label="Shared dynamics"),
}
ORDER = ["full", "tp1", "grm", "shared"]
EXCLUDE = {"b": {"shared"}}     # const/shared 的算子无方差 -> SNP 可预测性无定义


def cum_energy(K):
    """K: (N, P, P). 逐个体累积谱能量份额 -> (N, P)."""
    sv = np.linalg.svd(K, compute_uv=False)
    esq = sv ** 2
    return np.cumsum(esq, axis=1) / esq.sum(1)[:, None]


def add(rows, panel, series, x, mean, sd=np.nan):
    rows.append(dict(panel=panel, dataset=DS, series=series, x=int(x),
                     mean=float(mean), sd=float(sd) if sd is not None and np.isfinite(sd) else np.nan))


def build_data():
    if not os.path.isfile(AGG):
        raise SystemExit(f"缺少 {AGG}\n先跑 rank sweep:  python reproduce/arab/knode_rank_genetics_value_arab.py")
    agg = pd.read_csv(AGG)
    rows = []
    for _, r in agg.iterrows():
        s = COND2SERIES.get(r["cond"])
        if s is None:
            continue
        add(rows, "a", s, r["rank"], r["pcc_mean"], r.get("pcc_std"))
        add(rows, "b", s, r["rank"], r["ext_A_mean"], r.get("ext_A_std"))

    # ---- panel c: 优先读现成的 energy csv (本地画图用); 没有才从 max-rank cache 现算 (服务器上) ----
    if os.path.isfile(ENERG):
        en = pd.read_csv(ENERG)
        print(f"[panel c] 读现成 {ENERG} (rows={len(en)})")
    else:
        files = sorted(glob.glob(os.path.join(CACHE, f"f*_*_r{MAXR}_i*_ep*.npz")))
        if not files:
            raise SystemExit(f"既没有 {ENERG}, 也没有 r{MAXR} 的 cache: {CACHE}/f*_*_r{MAXR}_i*_ep*.npz\n"
                             f"先跑 rank sweep (RANKS 里含 {MAXR}), 或把服务器的 energy_maxrank.csv 拷来当 {os.path.basename(ENERG)}")
        print(f"[panel c] 没有 {os.path.basename(ENERG)}, 用 {len(files)} 个 r{MAXR} 的 npz 现算谱能量")
        recs = []
        for f in files:
            b = os.path.basename(f)
            # 文件名: f{fold}_{cond}_r{rank}_i{inner}_ep{epochs}.npz
            cond = b.split("_", 1)[1].split("_r")[0]
            z = np.load(f, allow_pickle=True)
            K = (z["U"] @ z["V"]).astype(np.float32)
            ce = cum_energy(K).mean(0)             # 对个体平均
            for k in KS:
                recs.append(dict(cond=cond, k=k, energy=float(ce[k - 1])))
        en = pd.DataFrame(recs).groupby(["cond", "k"], as_index=False)["energy"].mean()   # 对 fold x inner 平均
        en.to_csv(ENERG, index=False)
        print(f"[panel c] 现算并保存 -> {ENERG}")
    for _, r in en.iterrows():
        s = COND2SERIES.get(r["cond"])
        if s is not None:
            add(rows, "c", s, r["k"], r["energy"], None)

    d = pd.DataFrame(rows)
    d.to_csv(DATA, index=False)
    print(f"saved -> {DATA} | rows = {len(d)}")
    print(d.groupby(["panel", "series"]).size().to_string())
    return d


def get(d, panel, s):
    sub = d[(d.panel == panel) & (d.series == s)].sort_values("x")
    return sub["x"].values, sub["mean"].values, sub["sd"].values


def draw(ax, d, panel, with_band=True):
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
        ln, = ax.plot(x, y, marker=st["marker"], ms=st["ms"], lw=st["lw"], ls=st["ls"],
                      color=st["color"], zorder=3)
        handles.append(ln)
    return handles


def main():
    d = build_data()
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = GridSpec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.20, hspace=0.38)
    ax1 = fig.add_subplot(gs[:, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[1, 1])

    # ---- a ----
    h = draw(ax1, d, "a")
    ax1.axvline(2, ls="--", lw=1.4, color="gray", alpha=0.85, zorder=1)
    ax1.set_title("Trajectory prediction saturates at low rank", loc="left", fontsize=14,
                  fontweight="bold", pad=10)
    ax1.set_xlabel(r"Operator rank ($r$)", fontsize=12); ax1.set_ylabel("Recursive PCC", fontsize=12)
    xr = get(d, "a", "full")[0]
    ax1.set_xlim(0.5, float(xr.max()) + 3)
    vals = [v for s in ORDER if s not in EXCLUDE.get("a", set()) for v in get(d, "a", s)[1] if np.isfinite(v)]
    lo, hi = min(vals) - 0.025, max(vals) + 0.025
    ax1.set_ylim(lo, hi)
    ax1.text(2.8, hi - 0.005, r"$r=2$" + "\n(selected)", fontsize=10.5, va="top")
    ax1.set_xticks(xr)
    ax1.legend(handles=h, labels=[SERIES_STYLE[s]["label"] for s in ORDER
               if len(get(d, "a", s)[0])], frameon=False, fontsize=10.5, loc="lower right")

    # ---- b ----
    h = draw(ax2, d, "b")
    ax2.set_title("Operator predictability vs. rank", loc="left", fontsize=14, fontweight="bold", pad=10)
    ax2.set_xlabel(r"Operator rank ($r$)", fontsize=12)
    ax2.set_ylabel(r"SNP predictability of $A$ (r)", fontsize=12)
    ax2.set_xlim(0.5, float(xr.max()) + 3)
    bvals = [v for s in ORDER if s not in EXCLUDE.get("b", set()) for v in get(d, "b", s)[1] if np.isfinite(v)]
    ax2.set_ylim(min(bvals) - 0.05, max(bvals) + 0.05)
    ax2.set_xticks(xr)
    ax2.legend(handles=h, labels=[SERIES_STYLE[s]["label"] for s in ORDER
               if s not in EXCLUDE.get("b", set()) and len(get(d, "b", s)[0])],
               frameon=False, fontsize=10.5, loc="center right")

    # ---- c ----
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
    ax3.set_title("Operator spectrum is low-dimensional", loc="left", fontsize=14,
                  fontweight="bold", pad=10)
    ax3.set_xlabel(r"Number of singular modes ($k$)", fontsize=12)
    ax3.set_ylabel("Cumulative spectral energy", fontsize=12)
    # y 轴自适应: 按 x 显示范围内的所有点定下限, 避免低 k 的点被裁掉
    _vals = []
    for _s in ORDER:
        _xs, _ys, _sd = get(d, "c", _s)
        _vals += [float(_v) for _xx, _v in zip(_xs, _ys) if _xx <= 12 and np.isfinite(_v)]
    ax3.set_xlim(0.5, 12); ax3.set_ylim(min(_vals) - 0.05, 1.02)
    ax3.set_xticks([1, 2, 3, 4, 5, 8, 11])
    ax3.legend(handles=h, labels=[SERIES_STYLE[s]["label"] for s in ORDER
               if len(get(d, "c", s)[0])], frameon=False, fontsize=10.5, loc="lower right")

    for ax, lab in [(ax1, "a"), (ax2, "b"), (ax3, "c")]:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontweight("bold")
        ax.text(-0.13, 1.03, lab, transform=ax.transAxes, fontsize=20, fontweight="bold")

    fig.tight_layout()
    for ext in ("svg",):        # 只出 SVG
        out = os.path.join(HERE, f"fig_rank_rice.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("saved ->", out)
    plt.close(fig)


if __name__ == "__main__":
    print("run dir ->", RUN)
    main()
