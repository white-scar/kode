# -*- coding: utf-8 -*-
"""make_table1_data.py
从原始 per-fold 结果汇总出 Table 1 的数据（长表）。

输入 (原始结果, 在 outputs/reproduce/author_tp1_perfold/):
    {ds}_author_A_stats.csv        作者 A: A_h2_tr / A_h2_val
    {ds}_author_A_innerens.csv     作者 A: A_cv(交叉验证) / A_val(验证)   <- inner-ens 口径
    {ds}_kode_A_sig_trainval.csv   Kode  A: A_h2_tr / A_h2_val / A_cv / A_ext
    (文件里的 sig_* 列 = 奇异值, 本表不使用, 不写入)

输出 (本目录):
    table1_data.csv   列: dataset, method, metric, split, fold, value
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.normpath(os.path.join(HERE, "..", "..", "..", "outputs", "reproduce", "author_tp1_perfold"))
OUT  = os.path.join(HERE, "table1_data.csv")
print("raw dir ->", RAW)
assert os.path.isdir(RAW), f"原始目录不存在: {RAW}"

rows = []
for ds in ["maize", "arab"]:
    A  = pd.read_csv(os.path.join(RAW, f"{ds}_author_A_stats.csv"))
    Ai = pd.read_csv(os.path.join(RAW, f"{ds}_author_A_innerens.csv"))
    K  = pd.read_csv(os.path.join(RAW, f"{ds}_kode_A_sig_trainval.csv"))
    assert len(A) == len(Ai) == len(K) == 10, f"{ds}: 折数不是 10"

    def add(method, metric, split, vals):
        for i, v in enumerate(vals, 1):
            rows.append(dict(dataset=ds, method=method, metric=metric,
                             split=split, fold=i, value=float(v)))

    # dynamicGP (作者) —— A 由该基因型自身轨迹的 DMD 拟合
    add("author", "h2",             "train",            A["A_h2_tr"].values)
    add("author", "h2",             "validation",       A["A_h2_val"].values)
    add("author", "predictability", "cross-validation", Ai["A_cv"].values)
    add("author", "predictability", "validation",       Ai["A_val"].values)
    # Kode —— A 由超网络从 [GRM, y1] 生成
    add("kode",   "h2",             "train",            K["A_h2_tr"].values)
    add("kode",   "h2",             "validation",       K["A_h2_val"].values)
    add("kode",   "predictability", "cross-validation", K["A_cv"].values)
    add("kode",   "predictability", "validation",       K["A_ext"].values)

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print("saved ->", OUT, "| rows =", len(out))
print(out.groupby(["dataset", "method", "metric"]).agg(
      n=("value", "size"), mean=("value", "mean")).round(3).to_string())
