# -*- coding: utf-8 -*-
"""新旧冒烟数值逐列对比 (必须完全一致)"""
import pandas as pd
from pathlib import Path
R = Path(r"D:\pythonproject\knode\outputs\reproduce")
pairs = [
    ("maize", R/"knode_extrap_maize_nt5"/"knode_extrap_maize__smoke_overall.csv",
              R/"knode_extrap_maize_nt5"/"smoke"/"knode_extrap_maize_v2_f1_nt5_overall.csv"),
    ("arab",  R/"knode_extrap_arab_nt2"/"knode_extrap_arab__smoke_overall.csv",
              R/"knode_extrap_arab_nt2"/"smoke"/"knode_extrap_arab_v2_f1_nt2_overall.csv"),
]
cols = ["pcc", "pcc_snapshot", "pcc_pooled", "mse", "nmse", "nrmse", "r2"]
for name, fa, fb in pairs:
    a = pd.read_csv(fa).set_index("scenario")[cols]
    b = pd.read_csv(fb).set_index("scenario")[cols]
    d = (a - b).abs().max().max()
    same = a.sort_index().round(12).equals(b.sort_index().round(12))
    print(f"{name}: 旧 {fa.name}  新 {fb.name}")
    print(f"   max|diff| = {d:.3e}   完全一致 = {same}")
    if not same:
        print(a); print(b)
print("OK")
