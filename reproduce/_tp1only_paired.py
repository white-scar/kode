# -*- coding: utf-8 -*-
"""_tp1only_paired.py — 单性状 vs 全性状：逐 (fam, iteration) 配对差，判断是否稳定"""
import pandas as pd, numpy as np
BASE = "/data/fandata/dashuaige/outputs/reproduce"
JOBS = {
 "maize": [("E1","knode_tp1only_maize/all"), ("E2_single","knode_tp1only_maize/single")],
 "arab":  [("E1","knode_tp1only_arab/all"), ("E2_compactness01","knode_tp1only_arab/single_compactness01_visible"),
           ("E2_borderlength","knode_tp1only_arab/single_borderlength_fluo"), ("E2_leafwidth","knode_tp1only_arab/single_leafwidth_fluo")],
 "rice":  [("E1","knode_tp1only_rice/all"), ("E2_PSA","knode_tp1only_rice/single_PSA"),
           ("E2_TV_Compactness","knode_tp1only_rice/single_TV_Compactness"), ("E2_SV1_MaxDist","knode_tp1only_rice/single_SV1_MaxDist")],
 "uav":   [("E1","knode_tp1only_uav/all"), ("E2_TGI","knode_tp1only_uav/single_TGI"),
           ("E2_GCC","knode_tp1only_uav/single_GCC"), ("E2_RGBVI","knode_tp1only_uav/single_RGBVI")],
}
print("%-6s %-18s %-24s | %-8s %-8s | %-22s | %s" % ("ds","mode","trait","E1 PCC","E2 PCC","Δ per-iteration","同号"))
print("-"*140)
for ds, jobs in JOBS.items():
    tabs = {n: pd.read_csv("%s/%s/metrics_by_trait_time_per_iter.csv" % (BASE, r)) for n, r in jobs}
    e1 = tabs["E1"]
    for name, _ in jobs:
        if name == "E1": continue
        t = tabs[name]
        for tr in sorted(t.trait.unique()):
            a = e1[e1.trait == tr].groupby(["fam","iteration"]).pcc.mean()
            b = t[t.trait == tr].groupby(["fam","iteration"]).pcc.mean()
            d = (b - a).dropna()
            per_it = d.groupby("iteration").mean()
            frac = float((d > 0).mean())
            print("%-6s %-18s %-24s | %8.4f %8.4f | %-22s | %.0f%% (%d/%d)" % (
                ds, name, tr[:24], a.mean(), b.mean(),
                " ".join("%+.3f" % v for v in per_it.values), frac*100, int((d>0).sum()), len(d)))
