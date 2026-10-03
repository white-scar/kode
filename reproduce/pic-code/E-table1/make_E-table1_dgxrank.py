# -*- coding: utf-8 -*-
"""make_E-table1_dgxrank.py
由 E-table1_dgxrank_data.csv 生成 Supplementary Table 3 (宽表, 均值 ± SD, n=10 个 fold-set)。

输入: E-table1_dgxrank_data.csv   (长表, 每个 (dataset, D_G, rank, fold) 一行)
输出: E-table1_dgxrank_wide.csv        行 = (dataset, D_G, rank), 指标写成 "mean ± SD"

说明 (可写进表注):
  * 模型: Kode, full 条件 (GRM PCA + TP1), affine drift, 无 skip, 5 个 inner model 集成;
    指标排除 t1, 在训练折 per-trait min-max 归一化空间内计算。
  * val = 每个 fold-set 内 5 个 inner 模型对各自留出验证个体的 out-of-fold 预测;
    test = 外层 fold6 上的预测。
  * locked_default 标的是本项目 2026-09-19 锁定的默认配置 (非程序自动挑选结果):
    Arabidopsis D_G=30/r=2, Rice D_G=10/r=2, Maize RGB(drought) D_G=50/r=2。
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "E-table1_dgxrank_data.csv"))

LOCKED = {("Arabidopsis", 30, 2), ("Rice", 10, 2), ("Maize RGB", 50, 2)}

def ms(v):
    return "%.4f ± %.4f" % (v.mean(), v.std(ddof=1))

rows = []
for (ds, env, dg, rk), g in d.groupby(["dataset", "environment", "D_G", "rank"], sort=True):
    assert len(g) == 10, f"{ds}/{dg}/r{rk}: 折数不是 10 (={len(g)})"
    rows.append({
        "dataset": ds, "environment": env, "D_G": int(dg), "rank": int(rk),
        "val_PCC": ms(g["val_pcc"]), "val_MSE": ms(g["val_mse"]),
        "test_PCC": ms(g["test_pcc"]), "test_MSE": ms(g["test_mse"]),
        "n_foldsets": int(len(g)),
        "locked_default": "yes" if (ds, int(dg), int(rk)) in LOCKED else "",
    })

out = pd.DataFrame(rows).sort_values(["dataset", "D_G", "rank"])
out.to_csv(os.path.join(HERE, "E-table1_dgxrank_wide.csv"), index=False)
print(out.to_string(index=False))
print("\nsaved -> E-table1_dgxrank_wide.csv   rows =", len(out))
