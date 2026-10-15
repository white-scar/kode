# -*- coding: utf-8 -*-
"""_rank_paired_all.py -- 四个数据集: rank×cond 相对 full@r2 的逐 foldset 配对检验
单位: fold-set (n=10); percentile bootstrap 10000 次 -> 95% CI + 双侧 p; 每个数据集内 BH-FDR。
数据: 玉米 = 旧式脚本逐折 CSV (f{fold}.csv, 一行一个 (cond,rank));
      arab/rice/uav = rank_sweep_<ds>_dg<D>/cells/f{fold}_{cond}_r{rank}_ep1000.csv
输出: outputs/stats/rank_cond_paired_vs_full_r2.csv
"""
import os, glob, argparse
import numpy as np, pandas as pd

RD = "/root/autodl-tmp/qkn/outputs/reproduce"
MAIZE_LEGACY = "/tmp/maize_rankcond"
SPECS = [("maize", None,                  [1, 2, 3, 4, 6, 8, 10, 16, 20, 30, 50], 50),
         ("arab",  "rank_sweep_arab_dg30", [1, 2, 3, 4, 6, 8, 10, 16, 20, 30, 45], 45),
         ("rice",  "rank_sweep_rice_dg10", [1, 2, 3, 4, 6, 8, 10, 16, 20], 20),
         ("uav",   "rank_sweep_uav_dg50",  [1, 2, 3, 4, 6, 8, 10, 16], 16)]


def load(ds, sub, ranks):
    """-> dict[(cond, rank)] = DataFrame(fold, pcc, mse)"""
    out = {}
    if sub is None:                      # maize: 旧式逐折 CSV
        for f in range(10):
            p = os.path.join(MAIZE_LEGACY, "f%d.csv" % f)
            if not os.path.isfile(p):
                p = os.path.join(MAIZE_LEGACY, "knode_rank_genetics_value_f%d.csv" % f)
            d = pd.read_csv(p); d["fold"] = f
            for _, r in d.iterrows():
                out.setdefault((r["cond"], int(r["rank"])), []).append((f, r["pcc"], r["mse"]))
    else:
        for f in range(10):
            for cond in ("full", "tp1only", "grmonly", "const"):
                for rk in ranks:
                    p = os.path.join(RD, sub, "cells", "f%d_%s_r%d_ep1000.csv" % (f, cond, rk))
                    if not os.path.isfile(p):
                        continue
                    row = pd.read_csv(p).iloc[0]
                    out.setdefault((cond, int(rk)), []).append((f, row["pcc"], row["mse"]))
    return {k: pd.DataFrame(v, columns=["fold", "pcc", "mse"]).sort_values("fold").reset_index(drop=True)
            for k, v in out.items()}


def boot_p(d, n_boot=10000, seed=42):
    d = np.asarray(d, float); n = len(d)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return d.mean(), lo, hi, max(p, 1.0 / n_boot)


rows = []
for ds, sub, ranks, rmax in SPECS:
    D = load(ds, sub, ranks)
    if ("full", 2) not in D:
        print("[skip]", ds); continue
    base = D[("full", 2)]
    contrasts = [("full", 1), ("full", 3), ("full", 4), ("full", 8), ("full", rmax),
                 ("tp1only", 2), ("tp1only", 3), ("tp1only", 4), ("tp1only", 8), ("tp1only", rmax),
                 ("grmonly", 2), ("const", 2)]
    seen = set()
    for cond, rk in contrasts:
        if (cond, rk) in seen:
            continue
        seen.add((cond, rk))
        k = (cond, rk)
        if k not in D:
            continue
        o = D[k]
        m = base.merge(o, on="fold", suffixes=("_b", "_o"))
        dp = (m["pcc_o"] - m["pcc_b"]).values
        dm = (m["mse_o"] - m["mse_b"]).values
        mp, lo, hi, p = boot_p(dp)
        mm, mlo, mhi, pm = boot_p(dm)
        rows.append(dict(dataset=ds, cond=cond, rank=rk, n_fold=len(dp),
                         base="full@r2", base_pcc=base["pcc"].mean(),
                         pcc=o["pcc"].mean(), dpcc=mp, dpcc_lo=lo, dpcc_hi=hi, p_pcc=p,
                         dmse=mm, dmse_lo=mlo, dmse_hi=mhi, p_mse=pm))

res = pd.DataFrame(rows)
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q
res["q_pcc_fdr"] = res.groupby("dataset")["p_pcc"].transform(bh)
res["q_mse_fdr"] = res.groupby("dataset")["p_mse"].transform(bh)
res["sig_pcc"] = res["q_pcc_fdr"] < 0.05
res["sig_mse"] = res["q_mse_fdr"] < 0.05
os.makedirs("/root/autodl-tmp/qkn/outputs/stats", exist_ok=True)
out = "/root/autodl-tmp/qkn/outputs/stats/rank_cond_paired_vs_full_r2.csv"
res.to_csv(out, index=False)
print("saved ->", out, "rows =", len(res))
pd.set_option("display.width", 220)
for ds in res.dataset.unique():
    d = res[res.dataset == ds]
    print("\n===== %s (base full@r2 pcc=%.4f) =====" % (ds, d.base_pcc.iloc[0]))
    print(d[["cond", "rank", "pcc", "dpcc", "dpcc_lo", "dpcc_hi", "p_pcc", "q_pcc_fdr",
             "dmse", "q_mse_fdr", "sig_pcc"]].round(4).to_string(index=False))