# -*- coding: utf-8 -*-
"""baselines_per_trait_y.py - ? baselines_per_trait.py ??????????????/????

??: outputs/reproduce/knode_dynamic_baselines_full[_arab]_preds.npz  (key = {fam}_iter{k}_avg)
???:
  y_true = ? (fold, time, trait) ??????????????(?????)
  y_pred = ?????????
  ???? 0-9 ???? ? ? (fam, time, trait) ??????????
??? (pcc, mse, R2, NMSE, RMSE, NRMSE) ? baselines_per_trait.py ??????????

??: outputs/reproduce/knode_dynamic_baselines_full[_arab]_per_trait.csv
  ?: fam, iter, time, trait, pcc, mse, R2, NMSE, RMSE, NRMSE, y_true, y_pred
"""
import numpy as np, pandas as pd, sys
from pathlib import Path

BASE = Path("/data/fandata/dashuaige")
DS = sys.argv[1] if len(sys.argv) > 1 else "maize"
if DS == "maize":
    pheno = pd.read_csv(BASE / "data/maize_magic/maize_trait_BLUP_50_NArmv_traits_afterClustering.csv")
    cv    = pd.read_csv(BASE / "data/maize_magic/cross_validation_folds_w_validation_set_maize.csv", index_col=0)
    npz   = BASE / "outputs/reproduce/knode_dynamic_baselines_full_preds.npz"
    out   = BASE / "outputs/reproduce/knode_dynamic_baselines_full_per_trait.csv"
else:
    pheno = pd.read_csv(BASE / "data/arabidopsis/BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv")
    cv    = pd.read_csv(BASE / "data/arabidopsis/cross_validation_folds_w_validation_set_arabidopsis.csv", index_col=0)
    npz   = BASE / "outputs/reproduce/knode_dynamic_baselines_full_arab_preds.npz"
    out   = BASE / "outputs/reproduce/knode_dynamic_baselines_full_arab_per_trait.csv"

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
print("fams:", fams, flush=True)

rows = []
yt_acc = {}   # (fam, time, trait) -> [sum, count]
yp_acc = {}
for k in range(10):
    fc = cv.loc[gids].iloc[:, k].values.astype(int)
    tr = np.where((fc >= 1) & (fc <= 5))[0]; te = np.where(fc == 6)[0]
    tr_min = np.nanmin(Y[tr], axis=(0, 1)); tr_rng = np.maximum(np.nanmax(Y[tr], axis=(0, 1)) - tr_min, 1e-8)
    Yte = ((Y[te] - tr_min) / tr_rng)
    for fam in fams:
        key = f"{fam}_iter{k}_avg"
        if key not in z: continue
        pred = z[key]
        for t in range(1, T):
            for p in range(P):
                a = pred[:, t, p]; b = Yte[:, t, p]
                r = np.corrcoef(a, b)[0, 1] if (np.std(a) > 1e-12 and np.std(b) > 1e-12) else np.nan
                mse_ = float(np.mean((a - b) ** 2)); vy_ = float(np.var(b))
                kk = (fam, int(times[t]), traits[p])
                yt_acc.setdefault(kk, [0.0, 0]); yt_acc[kk][0] += float(np.mean(b)); yt_acc[kk][1] += 1
                yp_acc.setdefault(kk, [0.0, 0]); yp_acc[kk][0] += float(np.mean(a)); yp_acc[kk][1] += 1
                rows.append(dict(fam=fam, iter=k, time=int(times[t]), trait=traits[p],
                                 pcc=float(r), mse=mse_,
                                 R2=(1 - mse_ / vy_) if vy_ > 0 else np.nan,
                                 NMSE=(mse_ / vy_) if vy_ > 0 else np.nan,
                                 RMSE=float(np.sqrt(mse_)),
                                 NRMSE=(float(np.sqrt(mse_)) / abs(float(np.mean(b)))) if abs(float(np.mean(b))) > 1e-8 else np.nan))
    print(f"  iter {k} done", flush=True)

d = pd.DataFrame(rows)
d["y_true"] = [yt_acc[(f, t, tr)][0] / yt_acc[(f, t, tr)][1] for f, t, tr in zip(d.fam, d.time, d.trait)]
d["y_pred"] = [yp_acc[(f, t, tr)][0] / yp_acc[(f, t, tr)][1] for f, t, tr in zip(d.fam, d.time, d.trait)]
d = d[["fam","iter","time","trait","pcc","mse","R2","NMSE","RMSE","NRMSE","y_true","y_pred"]]
d.to_csv(out, index=False)
print("saved ->", out, "| rows =", len(d), "| cols =", list(d.columns), flush=True)
