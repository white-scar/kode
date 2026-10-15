# -*- coding: utf-8 -*-
"""_energy_from_cache.py -- 从 rank_sweep 的 max-rank cache 现算累积谱能量 (纯 CPU)
规则与玉米 panel c 完全一致: K_i = U_i V_i; energy(k) = sum_{j<=k} s_j^2 / sum_j s_j^2
先对个体平均, 再对 (fold x inner) 平均; KS 网格截断到各数据集的最大 rank (arab 45 / rice 20 / uav 16)。
输出: outputs/reproduce/<run_dir>/energy_maxrank.csv   (cond, k, energy)
"""
import os, glob
import numpy as np, pandas as pd

RDIR = "/root/autodl-tmp/qkn/outputs/reproduce"
# KS 用各数据集自己的 rank 网格 (各自封顶在 P), 与 fig_rank_*.py 里的 KS 一致
SPECS = [("arab", "rank_sweep_arab_dg30", 45, [1, 2, 3, 4, 6, 8, 10, 16, 20, 30, 45]),
         ("rice", "rank_sweep_rice_dg10", 20, [1, 2, 3, 4, 6, 8, 10, 16, 20]),
         ("uav",  "rank_sweep_uav_dg50", 16, [1, 2, 3, 4, 6, 8, 10, 16])]


def cum_energy(K):
    sv = np.linalg.svd(K, compute_uv=False)
    esq = sv ** 2
    return np.cumsum(esq, axis=1) / esq.sum(1)[:, None]


for name, sub, maxr, KS in SPECS:
    d = os.path.join(RDIR, sub, "cache")
    files = sorted(glob.glob(os.path.join(d, "f*_*_r%g_i*_ep*.npz" % maxr)))
    if not files:
        print("[skip] %s: 没有 r%g 的 cache" % (name, maxr)); continue
    recs = []
    for f in files:
        b = os.path.basename(f)
        cond = b.split("_", 1)[1].split("_r")[0]
        z = np.load(f, allow_pickle=True)
        K = (z["U"] @ z["V"]).astype(np.float32)
        ce = cum_energy(K).mean(0)
        for k in KS:
            recs.append(dict(cond=cond, k=k, energy=float(ce[k - 1])))
    en = pd.DataFrame(recs).groupby(["cond", "k"], as_index=False)["energy"].mean()
    out = os.path.join(RDIR, sub, "energy_maxrank.csv")
    en.to_csv(out, index=False)
    print("%-5s npz=%-4d conds=%s KS=%s -> %s rows=%d"
          % (name, len(files), sorted(en.cond.unique()), KS, out, len(en)))