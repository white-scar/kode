# -*- coding: utf-8 -*-
"""_author_rice_postproc_v2.py -- 照 author_rice_postproc.py, 路径换成新跑的位置, 输出 _v2
   输入: X_norm = author_tp1_rice/components/iter*_val_X_norm.csv
         enspred(R/phi) = author_all2_rice_innerens/components/iter*_val_{R,phi}_enspred_{tp1,mega}.csv
   输出: outputs/reproduce/author_rice_control_metrics_v2/author_rice_{long,by_time,overall}.csv
"""
import os
import numpy as np, pandas as pd
from scipy.stats import pearsonr
ROOT = Path = None
from pathlib import Path as _P
ROOT = _P("/root/autodl-tmp/qkn")
XN = ROOT / "outputs/reproduce/author_tp1_rice/components"
EN = ROOT / "outputs/reproduce/author_all2_rice_innerens/components"
OUT = ROOT / "outputs/reproduce/author_rice_control_metrics_v2"
OUT.mkdir(parents=True, exist_ok=True)
ph = pd.read_csv(ROOT / "data/rice_salinity/rice_indica_control_traits.csv")
ph = ph[ph["DAS"] >= 30]
ph["bio_ID"] = ph["bio_ID"].astype(str)
DAS = sorted(ph["DAS"].unique()); T = len(DAS)
print("n DAS:", T, DAS, flush=True)

def roll(A, X, mode):
    P, Tn = X.shape
    M = np.zeros_like(X, dtype=float); M[:, 0] = X[:, 0]
    for t in range(Tn - 1):
        M[:, t + 1] = A @ (X[:, t] if mode == "iter" else M[:, t])
    return M

rows, longs, diag = [], [], []
for k in range(1, 11):
    Xn = pd.read_csv(XN / ("iter%02d_val_X_norm.csv" % k), index_col=0)
    Xn.columns = Xn.columns.astype(str)
    gids = list(Xn.columns); n_tr = int(Xn.shape[0] / T)
    true = np.stack([Xn[g].values.reshape(n_tr, T).T for g in gids])
    for variant in ["tp1", "mega"]:
        Rr = pd.read_csv(EN / ("iter%02d_val_R_enspred_%s.csv" % (k, variant)), index_col=0)
        Phi = pd.read_csv(EN / ("iter%02d_val_phi_enspred_%s.csv" % (k, variant)), index_col=0)
        Rr.index = Rr.index.astype(str); Phi.index = Phi.index.astype(str)
        # R 的 make.names() 会把 "_-_-_" 里的 - 变成 . -> 实际列名分隔符可能是 _._._ ; 自动识别
        sep = "_-_-_" if any("_-_-_" in c for c in Phi.columns) else "_._._"
        traits = list(dict.fromkeys([c.split(sep)[0] for c in Phi.columns]))
        assert len(traits) == n_tr, (len(traits), n_tr)
        r = int(round(np.sqrt(Rr.shape[1])))
        for mode in ["iter", "rec"]:
            pred = np.full_like(true, np.nan)
            for gi, g in enumerate(gids):
                phi = np.array([[Phi.loc[g, tr + sep + "SV%d" % (s + 1)] for s in range(r)] for tr in traits])
                Rm = np.array([[Rr.loc[g, "SV%d%sSV%d" % (i + 1, sep, j + 1)] for j in range(r)] for i in range(r)])
                A = phi @ Rm @ np.linalg.pinv(phi)
                pred[gi] = roll(A, true[gi].T, mode).T
            for tpi, tp in enumerate(DAS):
                for pi in range(n_tr):
                    yp = pred[:, tpi, pi]; yt = true[:, tpi, pi]
                    m = ~np.isnan(yp) & ~np.isnan(yt)
                    if m.sum() < 3: continue
                    rr, _ = pearsonr(yt[m], yp[m])
                    ss_res = float(np.sum((yt[m] - yp[m]) ** 2)); ss_tot = float(np.sum((yt[m] - np.mean(yt[m])) ** 2))
                    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
                    mse = float(np.mean((yp[m] - yt[m]) ** 2)); rmse = float(np.sqrt(mse))
                    den = abs(float(np.mean(yt[m]))); nr = rmse / den if den > 1e-8 else np.nan
                    rows.append(dict(source="author", variant=variant, scenario=mode, foldset=k,
                                     timepoint=int(tp), trait=traits[pi],
                                     pcc=float(rr), R2=float(r2), MSE_norm=mse, NRMSE=float(nr)))
            for gi in range(len(gids)):
                pv = pred[gi].ravel(); tv = true[gi].ravel()
                m = ~np.isnan(pv) & ~np.isnan(tv)
                if m.sum() >= 3:
                    rr, _ = pearsonr(tv[m], pv[m])
                    if not np.isnan(rr):
                        longs.append(dict(variant=variant, scenario=mode, foldset=k, r_long=float(rr)))
            diag.append(dict(foldset=k, variant=variant, scenario=mode,
                             t1_pcc=float(pearsonr(true[:, 0, 0], pred[:, 0, 0])[0])))
    print("fold", k, "done", flush=True)

d = pd.DataFrame(rows)
d.to_csv(OUT / "author_rice_long.csv", index=False)
by_time = d.groupby(["variant","scenario","timepoint"]).agg(
    pcc=("pcc","mean"), pcc_std=("pcc","std"), R2=("R2","mean"), MSE_norm=("MSE_norm","mean"),
    MSE_norm_std=("MSE_norm","std"), NRMSE=("NRMSE","mean")).reset_index()
by_time.to_csv(OUT / "author_rice_by_time.csv", index=False)
lng = pd.DataFrame(longs)
ov = d.groupby(["variant","scenario"]).agg(pcc=("pcc","mean"), R2_median=("R2","median"),
        MSE_norm=("MSE_norm","mean"), NRMSE=("NRMSE","mean")).reset_index()
ov["r_long"] = [lng[(lng.variant==v)&(lng.scenario==s)].r_long.mean() for v, s in zip(ov.variant, ov.scenario)]
ov.to_csv(OUT / "author_rice_overall.csv", index=False)
print("\n=== author rice overall (13 点版) ==="); print(ov.round(4).to_string(index=False))
print("\n=== sanity: t1 PCC (应 ~1) ==="); print(pd.DataFrame(diag).head(4).to_string(index=False))