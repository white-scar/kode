# -*- coding: utf-8 -*-
"""_baselines_v2_tables.py -- 对今天重跑的基线 (_v2) 生成【逐时间点】和【逐 trait】表
逻辑照抄 reproduce/baselines_per_timepoint.py + baselines_per_trait.py (仅把数据源/输出名换成 _v2, 并支持 arab/rice/uav)
输出:
  outputs/reproduce/knode_dynamic_baselines_full_<ds>_v2_per_timepoint.csv  (fam, iter, time, pcc, mse)
  outputs/reproduce/knode_dynamic_baselines_full_<ds>_v2_per_trait.csv      (fam, iter, time, trait, pcc, mse, R2, NMSE, RMSE, NRMSE)
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path

BASE = Path("/root/autodl-tmp/qkn")
RD = BASE / "outputs" / "reproduce"
D = BASE / "data"


def load(ds):
    if ds == "arab":
        pheno = pd.read_csv(D / "arabidopsis" / "BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv")
        cv = pd.read_csv(D / "arabidopsis" / "cross_validation_folds_w_validation_set_arabidopsis.csv", index_col=0)
    elif ds == "rice":
        pheno = pd.read_csv(D / "rice_salinity" / "rice_indica_control_traits.csv", index_col=0)
        pheno = pheno[pheno["DAS"] >= 30].copy()          # 与基线 run 一致: 只用 DAS 30-42 (T=13)
        cv = pd.read_csv(D / "rice_salinity" / "rice_cv_10fold.csv", index_col=0)
        cv.index = cv.index.astype(str)
    elif ds == "uav":
        pheno = pd.read_csv(D / "maize_rgb" / "maize_rgb_drought_traits.csv")
        cv = pd.read_csv(D / "maize_rgb" / "maize_rgb_drought_cv_10fold.csv", index_col=0)
    else:
        raise SystemExit("unknown ds " + ds)
    cv.index = cv.index.astype(str)
    npz = RD / ("knode_dynamic_baselines_full_%s_v2_preds.npz" % ds)
    return pheno, cv, npz


def build(ds):
    pheno, cv, npz = load(ds)
    meta = ["bio_ID", "accession", "DAS"]
    traits = [c for c in pheno.columns if c not in meta and not c.startswith("Unnamed") and c != ""]
    traits = [t for t in traits if pheno[t].std() > 1e-12]
    times = sorted(pheno["DAS"].unique()); T = len(times); P = len(traits)
    gids = sorted(set(pheno["bio_ID"].astype(str)) & set(cv.index))
    Y = np.full((len(gids), T, P), np.nan)
    for pi, tr in enumerate(traits):
        sub = pheno[["bio_ID", "DAS", tr]].copy(); sub["bio_ID"] = sub["bio_ID"].astype(str)
        Y[:, :, pi] = sub.pivot(index="bio_ID", columns="DAS", values=tr).loc[gids].values
    z = np.load(npz)
    fams = sorted(set(k.split("_iter")[0] for k in z.keys()))
    print("%s: N=%d T=%d P=%d fams=%s" % (ds, len(gids), T, P, fams), flush=True)
    rows_tp, rows_tr = [], []
    for k in range(10):
        fc = cv.loc[gids].iloc[:, k].values.astype(int)
        tr = np.where((fc >= 1) & (fc <= 5))[0]; te = np.where(fc == 6)[0]
        tr_min = np.nanmin(Y[tr], axis=(0, 1)); tr_rng = np.maximum(np.nanmax(Y[tr], axis=(0, 1)) - tr_min, 1e-8)
        Yte = (Y[te] - tr_min) / tr_rng
        for fam in fams:
            key = f"{fam}_iter{k}_avg"
            if key not in z: continue
            pred = z[key]
            for t in range(1, T):
                rs = [np.corrcoef(pred[:, t, p], Yte[:, t, p])[0, 1] for p in range(P)
                      if np.std(pred[:, t, p]) > 1e-12 and np.std(Yte[:, t, p]) > 1e-12]
                mse = float(np.mean((pred[:, t, :] - Yte[:, t, :]) ** 2))
                rows_tp.append(dict(fam=fam, iter=k, time=int(times[t]), pcc=float(np.nanmean(rs)), mse=mse))
                for p in range(P):
                    a = pred[:, t, p]; b = Yte[:, t, p]
                    r = np.corrcoef(a, b)[0, 1] if (np.std(a) > 1e-12 and np.std(b) > 1e-12) else np.nan
                    mse_ = float(np.mean((a - b) ** 2)); vy_ = float(np.var(b)); mb = float(np.mean(b))
                    rows_tr.append(dict(fam=fam, iter=k, time=int(times[t]), trait=traits[p], pcc=float(r), mse=mse_,
                                        R2=(1 - mse_ / vy_) if vy_ > 0 else np.nan,
                                        NMSE=(mse_ / vy_) if vy_ > 0 else np.nan,
                                        RMSE=float(np.sqrt(mse_)),
                                        NRMSE=(float(np.sqrt(mse_)) / abs(mb)) if abs(mb) > 1e-8 else np.nan))
        print("  iter %d done" % k, flush=True)
    o1 = RD / ("knode_dynamic_baselines_full_%s_v2_per_timepoint.csv" % ds)
    o2 = RD / ("knode_dynamic_baselines_full_%s_v2_per_trait.csv" % ds)
    pd.DataFrame(rows_tp).to_csv(o1, index=False)
    pd.DataFrame(rows_tr).to_csv(o2, index=False)
    print("saved ->", o1, "rows", len(rows_tp))
    print("saved ->", o2, "rows", len(rows_tr))
    g = pd.DataFrame(rows_tp).groupby(["fam", "time"]).agg(pcc=("pcc", "mean"), mse=("mse", "mean")).reset_index()
    print(g.pivot(index="time", columns="fam", values="pcc").round(3).to_string())


for ds in (sys.argv[1:] or ["arab", "rice", "uav"]):
    build(ds)