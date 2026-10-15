# -*- coding: utf-8 -*-
"""_tp1only_common_traits.py — 在【共同性状】上比较 E1(全部TP1) vs E2(单性状) vs E3(少性状)"""
import pandas as pd
BASE = "/data/fandata/dashuaige/outputs/reproduce"
JOBS = {
 "maize": [("E1_all","knode_tp1only_maize/all"), ("E2_single","knode_tp1only_maize/single"), ("E3_few4","knode_tp1only_maize/few")],
 "arab":  [("E1_all","knode_tp1only_arab/all"), ("E2_compactness01","knode_tp1only_arab/single_compactness01_visible"),
           ("E2_borderlength","knode_tp1only_arab/single_borderlength_fluo"), ("E2_leafwidth","knode_tp1only_arab/single_leafwidth_fluo"),
           ("E3_few5","knode_tp1only_arab/few5")],
 "rice":  [("E1_all","knode_tp1only_rice/all"), ("E2_PSA","knode_tp1only_rice/single_PSA"),
           ("E2_TV_Compactness","knode_tp1only_rice/single_TV_Compactness"), ("E2_SV1_MaxDist","knode_tp1only_rice/single_SV1_MaxDist"),
           ("E3_few5","knode_tp1only_rice/few5")],
 "uav":   [("E1_all","knode_tp1only_uav/all"), ("E2_TGI","knode_tp1only_uav/single_TGI"), ("E2_GCC","knode_tp1only_uav/single_GCC"),
           ("E2_RGBVI","knode_tp1only_uav/single_RGBVI"), ("E3_few5","knode_tp1only_uav/few5")],
}
print("%-6s %-18s %4s | %-24s | %-24s | %s" % ("ds","mode","T","E1(同性状子集) PCC/MSE","该mode PCC/MSE","ΔPCC"))
print("-"*118)
for ds, jobs in JOBS.items():
    tabs = {n: pd.read_csv("%s/%s/metrics_by_trait_time.csv" % (BASE, r)) for n, r in jobs}
    e1 = tabs["E1_all"]
    for name, _ in jobs:
        if name == "E1_all": continue
        t = tabs[name]
        traits = sorted(t.trait.unique())
        s1 = e1[e1.trait.isin(traits)]
        a1 = s1.groupby("fam")[["pcc","mse"]].mean().mean()
        a2 = t.groupby("fam")[["pcc","mse"]].mean().mean()
        print("%-6s %-18s %4d | PCC %.4f  MSE %.5f      | PCC %.4f  MSE %.5f      | %+.4f" %
              (ds, name, len(traits), a1.pcc, a1.mse, a2.pcc, a2.mse, a2.pcc - a1.pcc))
