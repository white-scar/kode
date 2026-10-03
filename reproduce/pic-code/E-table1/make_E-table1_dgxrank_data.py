# -*- coding: utf-8 -*-
"""make_table3_data.py
把三个数据集的 D_G x rank 敏感性扫描结果汇总成长表 (Supplementary Table 3 的数据源)。

输入 (原始 per-fold 结果, 在 outputs/reproduce/dg_rank_{ds}/):
    f0.csv .. f9.csv   每个 (D_G, rank) cell 在每个 fold-set 上的
                       val_pcc / val_mse / test_pcc / test_mse
    (来自 knode_{arab,rice,rgb}_dg_rank_joint.py: full 条件, affine drift, 无 skip,
     5 个 inner model 集成, 指标排除 t1)

输出 (本目录):
    table3_data.csv   列: dataset, environment, D_G, rank, fold,
                           val_pcc, val_mse, test_pcc, test_mse, n_test
"""
import os, glob
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.normpath(os.path.join(HERE, "..", "..", "..", "outputs", "reproduce"))
OUT  = os.path.join(HERE, "E-table1_dgxrank_data.csv")
print("raw dir ->", RAW)

# 目录名 -> (表格里的数据集名, 环境)
DATASETS = {
    "dg_rank_arab": ("Arabidopsis", "control"),
    "dg_rank_rice": ("Rice", "control"),
    "dg_rank_rgb": ("Maize RGB", "drought"),
}

rows = []
for sub, (name, env) in DATASETS.items():
    d = os.path.join(RAW, sub)
    files = sorted(glob.glob(os.path.join(d, "f[0-9].csv")))
    if not files:
        print(f"[skip] {sub}: 没有 f*.csv")
        continue
    print(f"[read] {sub}: {len(files)} 个 fold 文件")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for _, r in df.iterrows():
        rows.append(dict(dataset=name, environment=env,
                         D_G=int(r["D_G"]), rank=int(r["rank"]), fold=int(r["fold"]),
                         val_pcc=float(r["val_pcc"]), val_mse=float(r["val_mse"]),
                         test_pcc=float(r["test_pcc"]), test_mse=float(r["test_mse"]),
                         n_test=int(r["n_test"])))

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print("saved ->", OUT, "| rows =", len(out))
print(out.groupby(["dataset", "environment"]).agg(
      cells=("D_G", "size"), folds=("fold", "nunique")).to_string())
