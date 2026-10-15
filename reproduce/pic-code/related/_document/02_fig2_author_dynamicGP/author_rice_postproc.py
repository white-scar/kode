# -*- coding: utf-8 -*-
"""author_rice_postproc.py - rebuild dynamicGP-MegaLMM(+TP1) rice predictions and compute
metrics in EXACTLY the same normalized space / same metric definitions as the Kode rice run."""
import os
import numpy as np, pandas as pd
from scipy.stats import pearsonr

ROOT = "/data/fandata/dashuaige"
COMP = f"{ROOT}/outputs/reproduce/author_dgp_rice_control/components"
OUT  = f"{ROOT}/outputs/reproduce/author_rice_control_metrics"
os.makedirs(OUT, exist_ok=True)

ph = pd.read_csv(f"{ROOT}/data/rice_salinity/rice_indica_control_traits.csv")
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
    Xn = pd.read_csv(f"{COMP}/iter{k:02d}_val_X_norm.csv", index_col=0)
    Xn.columns = Xn.columns.astype(str)
    gids = list(Xn.columns); n_tr = int(Xn.shape[0] / T)
    true = np.stack([Xn[g].values.reshape(n_tr, T).T for g in gids])      # (G,T,P) trait-major data
    for variant in ["tp1", "mega"]:
        Rr  = pd.read_csv(f"{COMP}/iter{k:02d}_val_R_enspred_{variant}.csv", index_col=0)
        Phi = pd.read_csv(f"{COMP}/iter{k:02d}_val_phi_enspred_{variant}.csv", index_col=0)
        Rr.index = Rr.index.astype(str); Phi.index = Phi.index.astype(str)
        traits = list(dict.fromkeys([c.split("_-_-_")[0] for c in Phi.columns]))
        assert len(traits) == n_tr, (len(traits), n_tr)
        r = int(round(np.sqrt(Rr.shape[1])))
        for mode in ["iter", "rec"]:
            pred = np.full_like(true, np.nan)
            for gi, g in enumerate(gids):
                phi = np.array([[Phi.loc[g, f"{tr}_-_-_SV{s+1}"] for s in range(r)] for tr in traits])
                Rm  = np.array([[Rr.loc[g, f"SV{i+1}_-_-_SV{j+1}"] for j in range(r)] for i in range(r)])
                A = phi @ Rm @ np.linalg.pinv(phi)
                pred[gi] = roll(A, true[gi].T, mode).T
            for tpi, tp in enumerate(DAS):
                for pi in range(n_tr):
                    yp = pred[:, tpi, pi]; yt = true[:, tpi, pi]
                    m = ~np.isnan(yp) & ~np.isnan(yt)
                    if m.sum() < 3: continue
                    rr, _ = pearsonr(yt[m], yp[m])
                    ss_res = float(np.sum((yt[m]-yp[m])**2)); ss_tot = float(np.sum((yt[m]-np.mean(yt[m]))**2))
                    r2 = 1 - ss_res/ss_tot if ss_tot > 1e-12 else np.nan
                    mse = float(np.mean((yp[m]-yt[m])**2)); rmse = float(np.sqrt(mse))
                    den = abs(float(np.mean(yt[m]))); nr = rmse/den if den > 1e-8 else np.nan
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
d.to_csv(f"{OUT}/author_rice_long.csv", index=False)
by_time = d.groupby(["variant","scenario","timepoint"]).agg(
    pcc=("pcc","mean"), pcc_std=("pcc","std"), R2=("R2","mean"), MSE_norm=("MSE_norm","mean"),
    MSE_norm_std=("MSE_norm","std"), NRMSE=("NRMSE","mean")).reset_index()
by_time.to_csv(f"{OUT}/author_rice_by_time.csv", index=False)
lng = pd.DataFrame(longs)
ov = d.groupby(["variant","scenario"]).agg(pcc=("pcc","mean"), R2_median=("R2","median"),
        MSE_norm=("MSE_norm","mean"), NRMSE=("NRMSE","mean")).reset_index()
ov["r_long"] = [lng[(lng.variant==v)&(lng.scenario==s)].r_long.mean() for v, s in zip(ov.variant, ov.scenario)]
ov.to_csv(f"{OUT}/author_rice_overall.csv", index=False)
print("\n=== author rice overall ==="); print(ov.round(4).to_string(index=False))
print("\n=== sanity: t1 PCC (should be ~1) ==="); print(pd.DataFrame(diag).head(4).to_string(index=False))
