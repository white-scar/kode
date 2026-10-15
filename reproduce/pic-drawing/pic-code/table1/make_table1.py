# -*- coding: utf-8 -*-
"""make_supp_table_operatorA.py
由图四的数据生成补充表格 (算子 A 的遗传力 / 可预测性)。

输入: table1_data.csv    长表 dataset, method, metric, split, fold, value
输出: supp_table_operatorA.csv   宽表 (均值 ± SD, n=10 折)

注意 (写进表注):
  Kode 的 operator A 是由模型从 [GRM, y1] 生成的, 因此其 "validation h2"
  衡量的是【生成算子与 GRM 的一致性】, 不是由表型拟合出的遗传力,
  不能与 dynamicGP 的 validation h2 (其 A 由验证集自身轨迹 DMD 拟合) 直接比较。
  因此本表以【算子可预测性 r(A_pred, A_true)】为主, h2 仅作描述性参考。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "table1_data.csv"))

SPLIT_LABEL = {"train": "training", "cross-validation": "cross-validation", "validation": "validation"}
rows = []
for ds in ["maize", "arab"]:
    for meth in ["author", "kode"]:
        sub = d[(d.dataset == ds) & (d.method == meth)]
        r = {"dataset": ds, "method": ("dynamicGP" if meth == "author" else "Kode")}
        for metric, mname in [("h2", "h2"), ("predictability", "r_pred")]:
            for split in ["train", "cross-validation", "validation"]:
                v = sub[(sub.metric == metric) & (sub.split == split)]["value"].values
                if len(v):
                    r[f"{mname}_{SPLIT_LABEL[split].replace('-', '_')}"] = "%.3f ± %.3f" % (v.mean(), v.std(ddof=1))
                else:
                    r[f"{mname}_{SPLIT_LABEL[split].replace('-', '_')}"] = ""
        rows.append(r)

out = pd.DataFrame(rows)
# 列排序
cols = ["dataset", "method",
        "h2_training", "h2_validation",
        "r_pred_cross_validation", "r_pred_validation"]
out = out[[c for c in cols if c in out.columns]]
out.to_csv(os.path.join(HERE, "table1.csv"), index=False)
print(out.to_string(index=False))
print("\nsaved -> table1.csv")
