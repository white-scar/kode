# -*- coding: utf-8 -*-
"""make_fig_baselines_data.py
图五数据：Kode 主方法 vs 动态基线，逐时间点 PCC / MSE。

输入 (outputs/):
  kode_result/maize/knode_tp1_{ode_hyper,iter_hyper}_per_trait.csv
  kode_result/arab/knode_arab_tp1_{ode_hyper,iter_hyper}_per_trait.csv
  reproduce/knode_dynamic_baselines_full_per_timepoint.csv
  reproduce/knode_dynamic_baselines_full_arab_per_timepoint.csv

输出 (本目录): fig_baselines_data.csv
  列: dataset, method, time, pcc, pcc_sd, mse, mse_sd
    Kode 侧: 先在每个 foldset 内对 trait 平均, 再对 10 个 foldset 求均值/SD
    基线侧: per_timepoint CSV 已是逐 iter 值, 直接对 10 个 iter 求均值/SD
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
OUT  = os.path.join(HERE, "fig_baselines_data.csv")

# 2026-09-27: ????? (Kode) ?????? per_trait run ?, ????? Fig.2 ?????,
# ???????/?/???; ??????????????????
# ?? (LSTM/AR-Transformer/ODE-RNN/GRU-ODE-Style/ST-STP) ????????? run?
UNIFIED = {
 "maize": os.path.join(ROOT, "reproduce","pic-code","pic2","unified_norm_author_vs_kode_per_foldset.csv"),
 "arab":  os.path.join(ROOT, "reproduce","pic-code","pic2","arab-E1","unified_norm_author_vs_kode_per_foldset.csv"),
 "rice":  os.path.join(ROOT, "reproduce","pic-code","pic2","rice-E2","unified_norm_author_vs_kode_per_foldset_v2.csv"),
}
# UAV ???????? (pic2 ?? uav), ???????? run
KODE_OWN = {
 "uav": (os.path.join(ROOT, "outputs", "kode_result", "maize_rgb_drought", "knode_maize_rgb_tp1_ode_hyper_hp_base_per_trait.csv"),
         os.path.join(ROOT, "outputs", "kode_result", "maize_rgb_drought", "knode_maize_rgb_tp1_iter_hyper_hp_base_per_trait.csv")),
}

BASE = {   # 2026-09-20: arab/rice/uav 用今天重跑的 _v2 基线
 "maize": os.path.join(ROOT, "outputs", "reproduce", "knode_dynamic_baselines_full_per_timepoint.csv"),
 "arab":  os.path.join(ROOT, "outputs", "reproduce", "knode_dynamic_baselines_full_arab_v2_per_timepoint.csv"),
 "rice":  os.path.join(ROOT, "outputs", "reproduce", "knode_dynamic_baselines_full_rice_v2_per_timepoint.csv"),
 "uav":   os.path.join(ROOT, "outputs", "reproduce", "knode_dynamic_baselines_full_uav_v2_per_timepoint.csv"),
}
BASE_TRAIT = {   # 逐 trait 明细 (箱线图用)
 "maize": os.path.join(ROOT, "outputs", "reproduce", "knode_dynamic_baselines_full_per_trait.csv"),
 "arab":  os.path.join(ROOT, "outputs", "reproduce", "knode_dynamic_baselines_full_arab_v2_per_trait.csv"),
 "rice":  os.path.join(ROOT, "outputs", "reproduce", "knode_dynamic_baselines_full_rice_v2_per_trait.csv"),
 "uav":   os.path.join(ROOT, "outputs", "reproduce", "knode_dynamic_baselines_full_uav_v2_per_trait.csv"),
}
OUT_LONG = os.path.join(HERE, "fig_baselines_mse_long.csv")

rows = []
for ds in ["maize", "arab", "rice", "uav"]:
    # ---- Kode (2026-09-14: SD 改为跨性状 SD — 每个性状先跨折平均, 再取性状间 SD) ----
    if ds in UNIFIED:
        # ---- Kode: ?????????? ----
        _u = pd.read_csv(UNIFIED[ds])
        for prot, ser in [("rec", "Kode_rec"), ("iter", "Kode_iter")]:
            d = _u[_u.Series == ser].rename(columns={"DAS": "timepoint", "PCC": "pcc", "MSE": "MSE_norm"})
            fv = d.groupby(["timepoint", "trait"], as_index=False)[["pcc", "MSE_norm"]].mean()
            g = fv.groupby("timepoint")[["pcc", "MSE_norm"]].agg(["mean", "std"])
            for t in g.index:
                rows.append(dict(dataset=ds, method=f"Kode_{prot}", time=int(t),
                                 pcc=float(g.loc[t, ("pcc", "mean")]),
                                 pcc_sd=float(g.loc[t, ("pcc", "std")]),
                                 mse=float(g.loc[t, ("MSE_norm", "mean")]),
                                 mse_sd=float(g.loc[t, ("MSE_norm", "std")])))
    else:
        # ---- UAV: ????, ????? run ----
        for prot, f in [("rec", KODE_OWN[ds][0]), ("iter", KODE_OWN[ds][1])]:
            d = pd.read_csv(f)
            d = d[(d["split"] == "test") & (d["timepoint"] != d["timepoint"].min())]
            fv = d.groupby(["timepoint", "trait"], as_index=False)[["pcc", "MSE_norm"]].mean()
            g = fv.groupby("timepoint")[["pcc", "MSE_norm"]].agg(["mean", "std"])
            for t in g.index:
                rows.append(dict(dataset=ds, method=f"Kode_{prot}", time=int(t),
                                 pcc=float(g.loc[t, ("pcc", "mean")]),
                                 pcc_sd=float(g.loc[t, ("pcc", "std")]),
                                 mse=float(g.loc[t, ("MSE_norm", "mean")]),
                                 mse_sd=float(g.loc[t, ("MSE_norm", "std")])))
    # ---- 基线 (同样用跨性状 SD; 均值与原来一致, 仅 SD 口径变化) ----
    b = pd.read_csv(BASE_TRAIT[ds])
    b = b[np.isfinite(b["pcc"])]
    b = b.groupby(["fam", "time", "trait"], as_index=False)[["pcc", "mse"]].mean()
    g = b.groupby(["fam", "time"])[["pcc", "mse"]].agg(["mean", "std"])
    for (fam, t), r in g.iterrows():
        rows.append(dict(dataset=ds, method=fam, time=int(t),
                         pcc=r[("pcc", "mean")], pcc_sd=r[("pcc", "std")],
                         mse=r[("mse", "mean")], mse_sd=r[("mse", "std")]))

# 2026-09-22: 论文统一称 "GRU-ODE-Style" (上游 fam key 仍是 gru_ode_bayes, 只在这里改显示名)
RENAME = {"gru_ode_bayes": "GRU-ODE-Style"}
out = pd.DataFrame(rows).sort_values(["dataset", "method", "time"])
out["method"] = out["method"].replace(RENAME)
# 2026-09-27: ???????? (pic4 ??: ?????, ????????)
DSDIR = {"maize": HERE, "arab": os.path.join(HERE, "arab"),
         "rice": os.path.join(HERE, "rice"), "uav": os.path.join(HERE, "uav")}
for _d in DSDIR.values():
    os.makedirs(_d, exist_ok=True)
for _ds, _d in DSDIR.items():
    _sub = out[out.dataset == _ds]
    _p = os.path.join(_d, "fig_baselines_data.csv")
    _sub.to_csv(_p, index=False)
    print("saved ->", _p, "| rows =", len(_sub))
print(out.groupby(["dataset", "method"]).size().to_string())

# ---------------- 逐 trait 的 MSE 长表 (箱线图用) ----------------
long_rows = []
for ds in ["maize", "arab", "rice", "uav"]:
    # Kode: per_trait.csv 里每个 (timepoint, trait, foldset) 一个 MSE_norm
    if ds in UNIFIED:
        _u = pd.read_csv(UNIFIED[ds])   # ??????????, ???????????
        for prot, ser in [("rec", "Kode_rec"), ("iter", "Kode_iter")]:
            d = _u[_u.Series == ser]
            long_rows.append(pd.DataFrame(dict(
                dataset=ds, method=f"Kode_{prot}",
                time=d["DAS"].astype(int).values, trait=d["trait"].values,
                fold=d["Fold"].astype(int).values, mse=d["MSE"].astype(float).values)))
    else:
        for prot, f in [("rec", KODE_OWN[ds][0]), ("iter", KODE_OWN[ds][1])]:
            d = pd.read_csv(f)
            d = d[(d["split"] == "test") & (d["timepoint"] != d["timepoint"].min())]
            long_rows.append(pd.DataFrame(dict(
                dataset=ds, method=f"Kode_{prot}",
                time=d["timepoint"].astype(int).values, trait=d["trait"].values,
                fold=d["foldset"].astype(int).values, mse=d["MSE_norm"].astype(float).values)))
    # 基线: per_trait.csv 里每个 (fam, iter, time, trait) 一个 mse
    b = pd.read_csv(BASE_TRAIT[ds])
    b = b[np.isfinite(b["mse"])]
    long_rows.append(pd.DataFrame(dict(
        dataset=ds, method=b["fam"].values, time=b["time"].astype(int).values,
        trait=b["trait"].values, fold=b["iter"].astype(int).values, mse=b["mse"].astype(float).values)))

L = pd.concat(long_rows, ignore_index=True)
L["method"] = L["method"].replace(RENAME)
for _ds, _d in DSDIR.items():
    _sub = L[L.dataset == _ds]
    _p = os.path.join(_d, "fig_baselines_mse_long.csv")
    _sub.to_csv(_p, index=False)
    print("saved ->", _p, "| rows =", len(_sub))
print(L.groupby(["dataset", "method"]).size().to_string())
