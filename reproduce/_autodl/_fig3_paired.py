# -*- coding: utf-8 -*-
"""Fig3 条件阶梯: full vs persistence/shared/grm/tp1 的逐 foldset 配对检验 (bootstrap 10000 + BH-FDR)"""
import os
import numpy as np, pandas as pd

RD = "/root/autodl-tmp/qkn/outputs/reproduce"
RUNS = {"arab": "cond_ladder_arab_dg30_r2", "rice": "cond_ladder_rice_dg10_r2", "uav": "cond_ladder_rgb_dg50_r2"}
CONDS = ["persistence", "shared", "grm", "tp1", "full"]


def boot_p(d, n_boot=10000, seed=42):
    d = np.asarray(d, float); n = len(d)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return d.mean(), lo, hi, max(p, 1.0 / n_boot)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q


rows = []
for ds, sub in RUNS.items():
    rec = {}
    for f in range(10):
        p = os.path.join(RD, sub, "f%d.csv" % f)
        if not os.path.isfile(p):
            print("[miss]", p); continue
        d = pd.read_csv(p); d["fold"] = f
        for _, r in d.iterrows():
            rec.setdefault(r["cond"], []).append((f, r["pcc"], r["mse"]))
    if "full" not in rec:
        print("[skip]", ds); continue
    B = pd.DataFrame(rec["full"], columns=["fold", "pcc", "mse"])
    for c in CONDS:
        if c == "full" or c not in rec:
            continue
        O = pd.DataFrame(rec[c], columns=["fold", "pcc", "mse"])
        m = B.merge(O, on="fold", suffixes=("_b", "_o"))
        dp = (m["pcc_o"] - m["pcc_b"]).values
        dm = (m["mse_o"] - m["mse_b"]).values
        mp, lo, hi, p = boot_p(dp); mm, mlo, mhi, pm = boot_p(dm)
        rows.append(dict(dataset=ds, contrast="%s - full" % c, n_fold=len(dp),
                         pcc=B["pcc"].mean(), pcc_cmp=O["pcc"].mean(),
                         dpcc=mp, dpcc_lo=lo, dpcc_hi=hi, p_pcc=p,
                         dmse=mm, dmse_lo=mlo, dmse_hi=mhi, p_mse=pm))
res = pd.DataFrame(rows)
res["q_pcc_fdr"] = res.groupby("dataset")["p_pcc"].transform(bh)
res["q_mse_fdr"] = res.groupby("dataset")["p_mse"].transform(bh)
res["sig_pcc"] = res["q_pcc_fdr"] < 0.05
os.makedirs("/root/autodl-tmp/qkn/outputs/stats", exist_ok=True)
out = "/root/autodl-tmp/qkn/outputs/stats/fig3_cond_ladder_paired_vs_full.csv"
res.to_csv(out, index=False)
print("saved ->", out, "rows =", len(res)); print()
pd.set_option("display.width", 200)
print(res[["dataset", "contrast", "pcc", "pcc_cmp", "dpcc", "dpcc_lo", "dpcc_hi", "p_pcc", "q_pcc_fdr",
           "dmse", "q_mse_fdr", "sig_pcc"]].round(4).to_string(index=False))