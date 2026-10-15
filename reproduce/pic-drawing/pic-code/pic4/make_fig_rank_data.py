# -*- coding: utf-8 -*-
"""make_fig_rank_data.py
把 rank 分析画图需要的所有数据汇总成一个 CSV（长表）。

输入 (每个数据集一个 rank sweep run dir, 见对应 knode_rank_genetics_value_*.py):
  outputs/reproduce/rank_sweep_<ds>_dg<D_G>/agg.csv   cond x rank 的 pcc_mean / ext_A_mean (→ panel a/b)
  pic4/fig_rank_energy.csv                            谱能量 (→ panel c, 目前只有玉米)

maize 例外 (2026-09-20): 玉米仍用老式脚本 reproduce/knode_rank_genetics_value.py 的产出
  outputs/reproduce/knode_rank_genetics_value_10fold_agg.csv  (dim=50 固定, rank 1..50, 5 cond)
  该表与凌晨 dg_rank 的 (D_G=50, rank=2) TEST PCC 0.512692 逐位一致, 故不再重跑 rank sweep。

输出 (本目录): fig_rank_data.csv
  列: panel, dataset, series, x, mean, sd
    panel=a : x=operator rank, series=full | tp1 | grm | shared
    panel=b : x=operator rank, series=full | tp1 | grm | shared   (算子 A 的 SNP 可预测性)
    panel=c : x=number of modes, series=full | tp1 | grm         (累积谱能量)
"""
import os, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.normpath(os.path.join(HERE, "..", "..", "..", "outputs", "reproduce"))
OUT  = os.path.join(HERE, "fig_rank_data.csv")

COND2SERIES = {"full": "full", "tp1only": "tp1", "grmonly": "grm", "const": "shared", "noise": "noise"}

rows = []
def add(panel, ds, series, x, mean, sd=np.nan):
    rows.append(dict(panel=panel, dataset=ds, series=series, x=int(x),
                     mean=float(mean), sd=float(sd) if sd is not None else np.nan))

# ---------- panel a: 轨迹 PCC ; panel b: 算子 A 的 SNP 可预测性 ----------
RUN_SUB = {"arab":          "rank_sweep_arab_dg30",
           "uav_drought":   "rank_sweep_uav_dg50",
           "uav_irrigated": "rank_sweep_uav_irrigated_dg50",
           "rice":          "rank_sweep_rice_dg10"}
# maize: 复用 09-18 老式脚本 (reproduce/knode_rank_genetics_value.py) 的 rank×cond 产出
MAIZE_AGG = os.path.join(RAW, "knode_rank_genetics_value_10fold_agg.csv")


def agg_path(ds):
    """返回该数据集的 rank×cond 汇总表路径 (schema 一致: cond, rank, pcc_mean/std, ext_A_mean/std, ...)。"""
    if ds == "maize":
        return MAIZE_AGG
    return os.path.join(RAW, RUN_SUB[ds], "agg.csv")


for ds in ["maize"] + list(RUN_SUB):
    agg = agg_path(ds)
    if not os.path.isfile(agg):
        print(f"[missing] {ds}: 还没有 {agg}, 先跑对应的 rank sweep")
        continue
    d = pd.read_csv(agg)
    for _, r in d.iterrows():
        s = COND2SERIES.get(r["cond"])
        if s is None:
            continue
        add("a", ds, s, r["rank"], r["pcc_mean"], r["pcc_std"])
        add("b", ds, s, r["rank"], r["ext_A_mean"], r["ext_A_std"])

# ---------- panel c: 累积谱能量 ----------
# 来源: fig_rank_energy.csv —— 由 rank sweep 的 cache 现算, 四个 cond 同源
ef = os.path.join(HERE, "fig_rank_energy.csv")
assert os.path.isfile(ef), f"缺少 {ef}"
en = pd.read_csv(ef)
for _, r in en.iterrows():
    s = COND2SERIES.get(r["cond"])
    if s is None:
        continue
    add("c", "maize", s, r["k"], r["energy"], np.nan)

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print("saved ->", OUT, "| rows =", len(out))
print(out.groupby(["panel", "dataset", "series"]).size().to_string())
