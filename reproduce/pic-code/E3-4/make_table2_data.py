# -*- coding: utf-8 -*-
"""make_table2_data.py -- 图7/图8 家族的统一数据构建 (玉米 MAGIC / 拟南芥 / 水稻)

逐行沿用旧脚本的聚合口径 (make_fig4_*_data.py 的 bytp/bytrait + make_fig5_sweep_data.py 的 sweep),
只把数据源换成当前最终结果的统一长表 pic-code/pic2/*/unified_norm_author_vs_kode_per_foldset.csv
(同一训练折 per-trait min-max 空间; Series = Kode_rec/Kode_iter/author_tp1_rec/author_tp1_iter/...).

输出 (reproduce/pic-code/table2/<ds>/):
  figE8_<ds>_pa_bytp_v4.csv       method, trait, tp, pcc          (tp = 真实 DAS, 剔首个时间点)
  figE8_<ds>_pa_bytrait_v4.csv    method, trait, pcc
  fig5_<ds>_sweep_data.csv        Threshold, Mean_Diff, SD_Diff, n_traits, Method
  h2_<ds>.csv                     Trait, Time(t<DAS>), Heritability
  rrblup_<ds>.csv                 Trait, Time, Accuracy
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))

DS = {
    "maize": dict(
        unified=os.path.join(ROOT, "reproduce", "pic-code", "pic2", "unified_norm_author_vs_kode_per_foldset.csv"),
        herit=os.path.join(HERE, "maize", "X_all_unrolled_h.csv"),
        rrblup=os.path.join(ROOT, "outputs", "dynamicGP", "Training_Validation_Results",
                            "rrBLUP_benchmarks_prediction_accuracies.csv"),
        thr_max=0.70, t1=15, label="Maize MAGIC"),
    "arab": dict(
        unified=os.path.join(ROOT, "reproduce", "pic-code", "pic2", "arab-E1", "unified_norm_author_vs_kode_per_foldset.csv"),
        herit=os.path.join(HERE, "arab", "X_all_unrolled_h.csv"),
        rrblup=os.path.join(ROOT, "outputs", "dynamicGP", "Arabidopsis", "Training_Validation_Results",
                            "rrBLUP_benchmarks_prediction_accuracies.csv"),
        thr_max=0.75, t1=8, label="Arabidopsis"),
    "rice": dict(
        unified=os.path.join(ROOT, "reproduce", "pic-code", "pic2", "rice-E2", "unified_norm_author_vs_kode_per_foldset_v2.csv"),
        herit=os.path.join(ROOT, "outputs", "reproduce", "rice_trait_selection_heritability_by_time.csv"),
        rrblup=os.path.join(ROOT, "outputs", "reproduce", "author_rice_control_metrics_rr",
                            "rrBLUP_benchmarks_prediction_accuracies_rice.csv"),
        thr_max=0.90, t1=28, label="Rice (indica, control)"),
}

MAP = {"Kode_rec": "tp1+knode_rec", "Kode_iter": "tp1+knode_iter",
       "author_tp1_rec": "tp1+Meg+Dgp_rec", "author_tp1_iter": "tp1+Meg+Dgp_iter"}
METHODS = ["tp1+Meg+Dgp_iter", "tp1+Meg+Dgp_rec", "tp1+knode_iter", "tp1+knode_rec"]

for ds, cfg in DS.items():
    out = os.path.join(HERE, ds)
    os.makedirs(out, exist_ok=True)
    u = pd.read_csv(cfg["unified"])
    u = u[u.Series.isin(MAP)].copy()
    u["method"] = u.Series.map(MAP)

    # ---- per time point / per trait (镜像 make_fig4_*_data.py) ----
    bytp = (u.groupby(["method", "trait", "DAS"], as_index=False)["PCC"].mean()
             .rename(columns={"DAS": "tp", "PCC": "pcc"}))
    bytp = bytp.sort_values(["method", "trait", "tp"]).reset_index(drop=True)
    bytrait = bytp.groupby(["method", "trait"], as_index=False)["pcc"].mean()
    bytrait = bytrait.sort_values(["method", "trait"]).reset_index(drop=True)
    bytp.to_csv(os.path.join(out, f"figE8_{ds}_pa_bytp_v4.csv"), index=False)
    bytrait.to_csv(os.path.join(out, f"figE8_{ds}_pa_bytrait_v4.csv"), index=False)

    # ---- rrBLUP benchmark -> sweep (镜像 make_fig5_sweep_data.py) ----
    rr = pd.read_csv(cfg["rrblup"])
    rr = rr.rename(columns={"Trait": "trait"})
    rr["Time"] = rr["Time"].astype(int)
    ex = cfg["t1"]
    rr_base = rr[rr["Time"] != ex].groupby("trait")["Accuracy"].mean()

    rows = []
    for method in METHODS:
        mt = bytrait[bytrait.method == method].set_index("trait")["pcc"]
        common = mt.index.intersection(rr_base.index)
        mt = mt[common]; base = rr_base[common]
        for th in [round(x, 2) for x in np.arange(0, cfg["thr_max"], 0.05)]:
            sel = mt >= th
            n = int(sel.sum())
            if n == 0:
                continue
            diff = mt[sel] - base[sel]
            rows.append(dict(Threshold=round(float(th), 2), Mean_Diff=float(diff.mean()),
                             SD_Diff=float(diff.std()), n_traits=n, Method=method))
    sw = pd.DataFrame(rows)
    sw.to_csv(os.path.join(out, f"fig5_{ds}_sweep_data.csv"), index=False)

    # ---- heritability (统一成 Trait / Time=t<DAS> / Heritability) ----
    h = pd.read_csv(cfg["herit"])
    if {"Trait", "Time", "Heritability"} <= set(h.columns):          # maize: 已是目标格式
        h = h[["Trait", "Time", "Heritability"]]
    elif {"Trait", "DAS", "H2"} <= set(h.columns):                    # rice: Trait/DAS/H2
        h = pd.DataFrame({"Trait": h["Trait"],
                          "Time": "t" + h["DAS"].astype(int).astype(str),
                          "Heritability": h["H2"]})
    else:                                                             # arab: key,Trait,Time,H2 (列名在 R 里被强制重命名)
        h = h.iloc[:, :4].copy(); h.columns = ["key", "Trait", "Time", "H2"]
        h = h.rename(columns={"H2": "Heritability"})[["Trait", "Time", "Heritability"]]
    h["Time"] = h["Time"].astype(str)
    h = h[h.Time.str.match(r"^t\d+$")]
    h.to_csv(os.path.join(out, f"h2_{ds}.csv"), index=False)

    # ---- rrBLUP benchmark 副本 (sweep 溯源用) ----
    rr[["trait", "Time", "Accuracy"]].rename(columns={"trait": "Trait"}).to_csv(
        os.path.join(out, f"rrblup_{ds}.csv"), index=False)

    print("=== %s (%s) ===" % (ds, cfg["label"]))
    print("  bytp", bytp.shape, "| bytrait", bytrait.shape, "| traits", bytrait.trait.nunique(),
          "| DAS", sorted(bytp.tp.unique()))
    print("  sweep", sw.shape, "| n_traits 范围", (sw.n_traits.min(), sw.n_traits.max()),
          "| rrblup∩kode traits", len(set(bytrait.trait) & set(rr_base.index)))
    print("  h2", h[["Trait", "Time", "Heritability"]].shape,
          "| H2 范围 %.3f..%.3f" % (h["Heritability"].min(), h["Heritability"].max()))
