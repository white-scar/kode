# -*- coding: utf-8 -*-
"""merge_rice_v2.py -- 图十 v2: Kode(13 点, 今天) vs 作者(13 点, 新跑)
Kode 侧来源: pic2/rice-E2/unified_norm_author_vs_kode_per_timepoint_v2.csv (逐点 mean/sd, 训练折 min-max 空间)
              + outputs/kode_result/rice_indica_control/knode_rice_tp1_all_summary_hp_base.csv (overall + r_long)
作者侧来源: author_rice_by_time_v2.csv / author_rice_overall_v2.csv (13 点版)
"""
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TP = ROOT / "outputs/kode_result/rice_indica_control/knode_rice_tp1_all_summary_hp_base.csv"
SRC = HERE.parent / "pic2" / "rice-E2" / "unified_norm_author_vs_kode_per_timepoint_v2.csv"

tab = pd.read_csv(SRC)
sm  = pd.read_csv(TP)
MAP = {"Kode_rec": "rec", "Kode_iter": "iter"}
bt, ov = [], []
for series, scen in MAP.items():
    for _, r in tab.iterrows():
        bt.append(pd.DataFrame([dict(method="Kode", scenario=scen, timepoint=int(r.DAS),
                       pcc=r[f"{series}_PCC"], pcc_sd=r[f"{series}_PCC_sd_fold"],
                       MSE_norm=r[f"{series}_MSE"], mse_sd=r[f"{series}_MSE_sd_fold"],
                       R2=r[f"{series}_R2"], NRMSE=float(np.sqrt(r[f"{series}_NMSE"])))]))
    m = "ode_hyper" if scen == "rec" else "iter_hyper"
    s = sm[(sm["mode"] == m) & (sm.split == "test")].iloc[0]
    ov.append(dict(method="Kode", scenario=scen, PCC=s.pcc, R2_median=s.R2_median,
                   MSE_norm=s.MSE_norm, NRMSE=s.NRMSE, r_long_flattened=s.r_long))
A = pd.read_csv(HERE / "author_rice_by_time_v2.csv")
for variant in ["tp1", "mega"]:
    x = A[A.variant == variant].copy()
    x["method"] = "dynamicGP-MegaLMM+TP1" if variant == "tp1" else "dynamicGP-MegaLMM"
    x = x.rename(columns={"MSE_norm": "MSE_norm", "MSE_norm_std": "mse_sd", "pcc_std": "pcc_sd"})
    bt.append(x[["method", "scenario", "timepoint", "pcc", "pcc_sd", "MSE_norm", "mse_sd", "R2", "NRMSE"]])
AO = pd.read_csv(HERE / "author_rice_overall_v2.csv")
for _, r in AO.iterrows():
    ov.append(dict(method=("dynamicGP-MegaLMM+TP1" if r.variant == "tp1" else "dynamicGP-MegaLMM"),
                   scenario=r.scenario, PCC=r.pcc, R2_median=r.R2_median,
                   MSE_norm=r.MSE_norm, NRMSE=r.NRMSE, r_long_flattened=r.r_long))
bt = pd.concat(bt, ignore_index=True).sort_values(["method", "scenario", "timepoint"])
bt.to_csv(HERE / "rice_control_bytime_v2.csv", index=False)
ov = pd.DataFrame(ov)
ov.to_csv(HERE / "rice_control_overall_v2.csv", index=False)
print("=== overall (13 点, 今天) ==="); print(ov.round(4).to_string(index=False))
print("\nbytime rows:", len(bt), "| DAS:", sorted(bt.timepoint.unique()))
