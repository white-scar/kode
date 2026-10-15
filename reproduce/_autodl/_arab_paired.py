# -*- coding: utf-8 -*-
"""拟南芥 rank sweep: 逐 foldset 配对比较 (full r2 vs 其它) —— 差距是否显著"""
import glob, os
import numpy as np, pandas as pd
from scipy import stats

CELLS = "/root/autodl-tmp/qkn/outputs/reproduce/rank_sweep_arab_dg30/cells"
def vec(cond, rank):
    rows = []
    for f in range(10):
        p = os.path.join(CELLS, "f%d_%s_r%d_ep1000.csv" % (f, cond, rank))
        if not os.path.isfile(p):
            return None
        rows.append(pd.read_csv(p).iloc[0])
    d = pd.DataFrame(rows)
    return d["pcc"].values.astype(float), d["mse"].values.astype(float)

base_p, base_m = vec("full", 2)
print("拟南芥 (n=10 foldset) 逐折配对比较, 基准 = full@r2 (pcc=%.4f±%.4f, mse=%.6f±%.6f)"
      % (base_p.mean(), base_p.std(ddof=1), base_m.mean(), base_m.std(ddof=1)))
print()
print("%-14s %-8s %8s %8s %9s %9s %9s" % ("对比", "对比均值", "Δpcc", "Δ标准误", "配对t p", "Wilcoxon p", "Δmse"))
for cond, rank in [("full",1),("full",3),("full",4),("full",8),("full",45),
                   ("tp1only",2),("tp1only",3),("tp1only",4),("tp1only",8),("tp1only",20),
                   ("grmonly",2),("const",2)]:
    v = vec(cond, rank)
    if v is None:
        print("%-14s  (缺)" % ("%s@r%d" % (cond, rank))); continue
    p, m = v
    dp = p - base_p
    t, pt = stats.ttest_rel(p, base_p)
    try:
        w, pw = stats.wilcoxon(p, base_p)
    except Exception:
        pw = np.nan
    print("%-14s %8.4f %+8.4f %9.4f %9.4f %9.4f %+9.5f"
          % ("%s@r%d" % (cond, rank), p.mean(), dp.mean(), dp.std(ddof=1)/np.sqrt(10), pt, pw, (m - base_m).mean()))
print()
print("full 各 rank 的逐折均值:", {r: round(vec("full", r)[0].mean(), 4) for r in [1,2,3,4,6,8,10,16,20,30,45]})
print("tp1only 各 rank 的逐折均值:", {r: round(vec("tp1only", r)[0].mean(), 4) for r in [1,2,3,4,6,8,10,16,20,30,45]})