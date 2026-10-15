# -*- coding: utf-8 -*-
"""merge rice in-window: Kode vs author dynamicGP-MegaLMM+TP1 (indica control, 10 fold-sets).
Both sides use the SAME train-fold per-trait min-max space, so MSE is directly comparable."""
import pandas as pd
kd = pd.read_csv("kode_rice_detail.csv")     # mode x split x timepoint (mean over 10 foldsets)
ks = pd.read_csv("kode_rice_summary.csv")    # mode x split overall
ad = pd.read_csv("author_rice_by_time.csv")  # variant x scenario x timepoint
ao = pd.read_csv("author_rice_overall.csv")

MAP_K = {"ode_hyper": "rec", "iter_hyper": "iter"}
bt = []
for mode, scen in MAP_K.items():
    x = kd[(kd["mode"] == mode) & (kd.split == "test")].copy()
    x["method"] = "Kode"; x["scenario"] = scen
    bt.append(x.rename(columns={"pcc_std": "pcc_sd", "MSE_norm_std": "mse_sd"})[
        ["method","scenario","timepoint","pcc","pcc_sd","MSE_norm","mse_sd","R2","NRMSE"]])
for variant in ["tp1","mega"]:
    x = ad[ad.variant == variant].copy()
    x["method"] = "dynamicGP-MegaLMM+TP1" if variant == "tp1" else "dynamicGP-MegaLMM"
    bt.append(x.rename(columns={"MSE_norm":"MSE_norm","MSE_norm_std":"mse_sd","pcc_std":"pcc_sd"})[
        ["method","scenario","timepoint","pcc","pcc_sd","MSE_norm","mse_sd","R2","NRMSE"]])
bt = pd.concat(bt, ignore_index=True).sort_values(["method","scenario","timepoint"])
bt.to_csv("rice_control_bytime.csv", index=False)

ov = []
for mode, scen in MAP_K.items():
    r = ks[(ks["mode"] == mode) & (ks.split == "test")].iloc[0]
    ov.append(dict(method="Kode", scenario=scen, PCC=r.pcc, R2_median=r.R2_median,
                   MSE_norm=r.MSE_norm, NRMSE=r.NRMSE, r_long_flattened=r.r_long))
for _, r in ao.iterrows():
    ov.append(dict(method=("dynamicGP-MegaLMM+TP1" if r.variant=="tp1" else "dynamicGP-MegaLMM"),
                   scenario=r.scenario, PCC=r.pcc, R2_median=r.R2_median,
                   MSE_norm=r.MSE_norm, NRMSE=r.NRMSE, r_long_flattened=r.r_long))
ov = pd.DataFrame(ov)
ov.to_csv("rice_control_overall.csv", index=False)
print("=== overall (test split, 10 fold-sets) ===")
print(ov.round(4).to_string(index=False))
print("\n=== by timepoint (first 6 rows) ===")
print(bt.head(6).round(4).to_string(index=False))
