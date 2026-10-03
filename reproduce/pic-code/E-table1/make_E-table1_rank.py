# -*- coding: utf-8 -*-
"""make_E-table1_rank.py
把四个数据集"锁定 D_G 下的 operator-rank 扫描"整理成 E-table1 (拓展表1)。
输入 (既有结果, 不重跑):
  outputs/reproduce/knode_rank_genetics_value_10fold_agg.csv   (Maize MAGIC, 旧脚本 knode_rank_genetics_value.py)
  outputs/reproduce/rank_sweep_arab_dg30/agg.csv               (Arabidopsis)
  outputs/reproduce/rank_sweep_rice_dg10/agg.csv               (Rice)
  outputs/reproduce/rank_sweep_uav_dg50/agg.csv                (Maize UAV)
  (每份 agg.csv 的列: cond, rank, ..., pcc_mean, pcc_std, mse_mean, mse_std, n; 这里取 cond == "full")
输出 (本目录):
  E-table1_rank_long.csv            长表 (带 SD, 备查)
  E-table1_rank_values.csv          简洁值表 dataset, D_G, rank, test_PCC, test_MSE (3 位小数)
  E-table1_rank_paste.md            可直接贴的三线表 (PCC/MSE 均 3 位小数) + 表注
  E-table1_rank_paste_altMSE4.md    备选 (PCC 3 位 + MSE 4 位)
  E-table1_rank_table.tex           booktabs 三线表 (3 位小数)
注: 玉米 r=2 的 PCC 按正文 (Fig.2) 口径写成 0.512。
"""
import os, io
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
R    = os.path.join(ROOT, "outputs", "reproduce")

SRC = [("Maize MAGIC", 50, "knode_rank_genetics_value_10fold_agg.csv"),
       ("Arabidopsis", 30, "rank_sweep_arab_dg30/agg.csv"),
       ("Rice",        10, "rank_sweep_rice_dg10/agg.csv"),
       ("Maize UAV",   50, "rank_sweep_uav_dg50/agg.csv")]
OVERRIDE = {("Maize MAGIC", 2, "pcc"): 0.512}

long_rows = []
for name, dg, f in SRC:
    d = pd.read_csv(os.path.join(R, f))
    d = d[d.cond == "full"].sort_values("rank")
    for _, r in d.iterrows():
        long_rows.append(dict(dataset=name, D_G_locked=dg, rank=int(r["rank"]),
                              pcc_mean=float(r.pcc_mean), pcc_sd=float(r.pcc_std),
                              mse_mean=float(r.mse_mean), mse_sd=float(r.mse_std), n=int(r["n"])))
L = pd.DataFrame(long_rows)
L.to_csv(os.path.join(HERE, "E-table1_rank_long.csv"), index=False)
ranks = sorted(L["rank"].unique())
DS_ORDER = [s[0] for s in SRC]

def val(ds, r, metric):
    key = (ds, r, metric)
    if key in OVERRIDE: return OVERRIDE[key]
    return float(L[(L.dataset == ds) & (L["rank"] == r)][metric + "_mean"].iloc[0])

rows = []
for ds in DS_ORDER:
    dg = int(L[L.dataset == ds].D_G_locked.iloc[0])
    for r in ranks:
        if r in set(L[L.dataset == ds]["rank"]):
            rows.append(dict(dataset=ds, D_G=dg, rank=r,
                             test_PCC=round(val(ds, r, "pcc"), 3), test_MSE=round(val(ds, r, "mse"), 3)))
pd.DataFrame(rows).to_csv(os.path.join(HERE, "E-table1_rank_values.csv"), index=False)

NOTE = ("表注: 在所有数据集上, operator rank >= 2 即进入平台 (rank >= 2 各档的极差: "
        "玉米 MAGIC PCC 0.011 / MSE 0.00026; 拟南芥 0.015 / 0.00041; 水稻 0.008 / 0.00021; "
        "玉米 UAV 0.020 / 0.00675), 继续增大 rank 不再带来一致的增益; 由于作者方法的算子秩固定为 2, "
        "为使两者在同一模型容量下可比, 后续全部分析统一采用 rank = 2。"
        "GRM PCA 维度取各数据集的锁定值 (玉米 MAGIC / 玉米 UAV = 50, 拟南芥 = 30, 水稻 = 10), "
        "该锁定值来自 D_G 的敏感性扫描。玉米 r=2 的 PCC 取主分析 (Fig. 2) 口径 0.512。"
        "模型为 Kode full 条件 (GRM PCA + TP1)、affine drift、无 skip、5 个 inner model 集成; "
        "指标排除 t1, 在训练折 per-trait min-max 归一化空间内计算; 数值为外层 fold6 上 10 个 fold-set 的均值。")

def tables(nd_pcc, nd_mse):
    md = []
    for metric, name, nd in (("pcc", "a. Test PCC", nd_pcc), ("mse", "b. Test MSE (normalized)", nd_mse)):
        md.append("## " + name)
        md.append("| Dataset | D_G | " + " | ".join("r=%d" % r for r in ranks) + " |")
        md.append("|" + "---|" * (len(ranks) + 2))
        for ds in DS_ORDER:
            dg = int(L[L.dataset == ds].D_G_locked.iloc[0])
            cells = ["%.*f" % (nd, val(ds, r, metric)) if r in set(L[L.dataset == ds]["rank"]) else "—" for r in ranks]
            md.append("| %s | %d | %s |" % (ds, dg, " | ".join(cells)))
        md.append("")
    return md

head = ["# E-table1 (拓展表1) — Kode 的 operator-rank 敏感性", "",
        "模型: Kode full 条件 (GRM PCA + TP1), affine drift, 无 skip, 5 个 inner model 集成; "
        "指标排除 t1, 训练折 per-trait min-max 归一化空间; 数值 = 外层 fold6 上 10 个 fold-set 的均值; "
        "各数据集用其锁定 D_G。", ""]
io.open(os.path.join(HERE, "E-table1_rank_paste.md"), "w", encoding="utf-8", newline="\n").write(
    "\n".join(head + tables(3, 3) + [NOTE, ""]))
io.open(os.path.join(HERE, "E-table1_rank_paste_altMSE4.md"), "w", encoding="utf-8", newline="\n").write(
    "\n".join(["# E-table1 (拓展表1, 备选: MSE 保留 4 位小数)", "", NOTE, ""] + tables(3, 4)))

def tex_table(cap, lab, metric, nd, lines):
    lines += [r"\begin{table}[ht]", r"\centering", r"\caption{%s}" % cap, r"\label{%s}" % lab, r"\small",
              r"\begin{tabular}{lrr" + "r" * (len(ranks) - 1) + "}", r"\toprule",
              "Dataset & $D_G$ & " + " & ".join(str(r) for r in ranks) + r" \\", r"\midrule"]
    for ds in DS_ORDER:
        dg = int(L[L.dataset == ds].D_G_locked.iloc[0])
        cells = ["%.*f" % (nd, val(ds, r, metric)) if r in set(L[L.dataset == ds]["rank"]) else "--" for r in ranks]
        lines.append("%s & %d & %s \\\\" % (ds, dg, " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

tex = ["% Extended Data Table 1 -- Kode operator-rank sensitivity (3 decimals; maize r=2 PCC from Fig.2).",
       "% " + NOTE, ""]
tex_table("Test-set prediction accuracy (PCC) as a function of operator rank.", "tab:etable1a", "pcc", 3, tex)
tex_table("Test-set normalized MSE as a function of operator rank.", "tab:etable1b", "mse", 3, tex)
io.open(os.path.join(HERE, "E-table1_rank_table.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(tex))
print("saved: E-table1_rank_{long,values}.csv / _paste.md / _paste_altMSE4.md / _table.tex | rows =", len(L))
