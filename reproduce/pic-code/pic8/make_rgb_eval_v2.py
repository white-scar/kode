# -*- coding: utf-8 -*-
"""make_rgb_eval.py -- Maize UAV RGB: build the pic2-style long table.

Follows pic2/unified_norm_eval.py:
   per_foldset  : Series, Fold, DAS, trait, PCC, MSE      (Series = method_protocol_env)
   per_timepoint: per DAS, for each Series: PCC/PCC_sd/PCC_sd_fold, MSE/MSE_sd/MSE_sd_fold
     pooled SD  = std over (trait x fold)   <- main
     _sd_fold   = per-fold mean over traits, then std over folds

Series:
   Kode_rec / Kode_iter            <- knode_maize_rgb_tp1_{ode_hyper,iter_hyper}_per_trait.csv
   baseline_rrBLUPt1 / baseline_t1 <- computed here (GBLUP-predicted t1 held / observed t1 held)
Normalization (same space as Kode, i.e. training-fold per-trait min-max):
   baselines are computed in the same train-fold min-max space.
"""
import os, numpy as np, pandas as pd
from scipy.stats import pearsonr

BASE = r"D:\pythonproject\knode"
DATA = os.path.join(BASE, "data", "maize_rgb")
OUT  = os.path.join(BASE, "reproduce", "pic-code", "pic8")
FL   = [42, 56, 61, 64, 68, 71, 75, 78, 82, 84, 90, 99]
os.makedirs(OUT, exist_ok=True)

grm_all = pd.read_csv(os.path.join(DATA, "geno_GRM.csv"), index_col=0)
grm_all.index = grm_all.index.astype(str); grm_all.columns = grm_all.columns.astype(str)

def gblup_t1(Y_tr_t1, g_tr, g_te):
    """per-trait GBLUP on t1 (REML grid for the variance ratio), return (n_te, P)."""
    w, U = np.linalg.eigh(g_tr); w = np.clip(w, 1e-8, None)
    n_tr, P = Y_tr_t1.shape
    rs = np.geomspace(1e-6, 1e6, 200); out = np.zeros((g_te.shape[0], P))
    for p in range(P):
        y = Y_tr_t1[:, p].astype(float); mu = y.mean(); yc = y - mu
        yt = U.T @ yc; s1 = U.sum(0); bL = -np.inf; bR = 1.0
        for r in rs:
            d = w + r; t1 = np.sum(s1 ** 2 / d)
            yPy = np.sum(yt ** 2 / d) - (np.sum(yt * s1 / d)) ** 2 / t1
            s2 = max(yPy / (n_tr - 1), 1e-12)
            logL = -0.5 * ((n_tr - 1) * np.log(s2) + np.sum(np.log(d)) + np.log(t1))
            if logL > bL: bL, bR = logL, r
        out[:, p] = g_te @ np.linalg.solve(g_tr + bR * np.eye(n_tr), yc) + mu
    return out

rows = []
for env in ["drought"]:      # v2: irrigated 在新流程里未跑 (无 rank sweep / hp / 基线), 先只做 drought
    # ---- traits / cv / grm ----
    d = pd.read_csv(os.path.join(DATA, f"maize_rgb_{env}_traits.csv"))   # v2: bio_ID 是列, 不能当 index
    d["bio_ID"] = d["bio_ID"].astype(str)
    cv = pd.read_csv(os.path.join(DATA, f"maize_rgb_{env}_cv_10fold.csv"), index_col=0)
    cv.index = cv.index.astype(str)
    traits = [c for c in d.columns if c not in ("bio_ID", "DAS")]
    gids = sorted(d["bio_ID"].unique())
    Y = np.full((len(gids), len(FL), len(traits)), np.nan)
    for pi, t in enumerate(traits):
        Y[:, :, pi] = d.pivot(index="bio_ID", columns="DAS", values=t).reindex(gids)[FL].values
    K = grm_all.loc[gids, gids].values.astype(float)

    # ---- Kode (read the template's per_trait) ----
    for mode, prot in [("ode_hyper", "rec"), ("iter_hyper", "iter")]:
        p = os.path.join(BASE, "outputs", "kode_result", f"maize_rgb_{env}",
                         f"knode_maize_rgb_tp1_{mode}_hp_base_per_trait.csv")   # v2: 今天的主方法结果
        x = pd.read_csv(p)
        x = x[(x.split == "test") & (x.timepoint != FL[0])]
        rows.append(pd.DataFrame(dict(Series=f"Kode_{prot}_{env}", Fold=x.foldset.values,
                                      DAS=x.timepoint.values, trait=x.trait.values,
                                      PCC=x.pcc.values, MSE=x.MSE_norm.values)))
        print(f"[{env}] Kode_{prot}: {len(x)} rows from {os.path.basename(p)}")

    # ---- baselines ----
    for k in range(10):
        fc = cv.loc[gids, f"V{k+1}"].values.astype(int)
        tri = np.where((fc >= 1) & (fc <= 5))[0]; tei = np.where(fc == 6)[0]
        tmn = np.nanmin(Y[tri], axis=(0, 1)); tmx = np.nanmax(Y[tri], axis=(0, 1))
        trg = np.maximum(tmx - tmn, 1e-8)
        Ys = (Y - tmn) / trg
        Yte = Ys[tei]
        # (a) GBLUP-predicted t1 held
        pred_g = gblup_t1(Ys[tri][:, 0, :], K[np.ix_(tri, tri)], K[np.ix_(tei, tri)])
        # (b) observed t1 held
        pred_o = Yte[:, 0, :]
        for tag, P_ in [("baseline_rrBLUPt1", pred_g), ("baseline_t1", pred_o)]:
            for ti, das in enumerate(FL):
                if ti == 0: continue
                for pj, tname in enumerate(traits):
                    a = P_[:, pj]; b = Yte[:, ti, pj]
                    pcc = pearsonr(a, b)[0] if (np.std(a) > 1e-12 and np.std(b) > 1e-12) else np.nan
                    rows.append(pd.DataFrame([dict(Series=f"{tag}_{env}", Fold=k + 1, DAS=das,
                                                   trait=tname, PCC=float(pcc),
                                                   MSE=float(np.mean((a - b) ** 2)))]))
        print(f"[{env}] baselines fold {k+1} done")

fold = pd.concat(rows, ignore_index=True)[["Series", "Fold", "DAS", "trait", "PCC", "MSE"]].sort_values(["Series", "DAS"])
fold.to_csv(os.path.join(OUT, "maize_rgb_per_foldset_v2.csv"), index=False)
print("\nfolded rows =", len(fold), " Series =", list(fold.Series.unique()))

# ---- per_timepoint (pic2 format) ----
SER = list(fold.Series.unique())
tab = pd.DataFrame({"DAS": [t for t in FL if t != FL[0]]})
for s in SER:
    sub = fold[fold.Series == s]
    pooled = sub.groupby("DAS")[["PCC", "MSE"]].agg(["mean", "std"])
    foldm  = sub.groupby(["DAS", "Fold"])[["PCC", "MSE"]].mean()
    foldsd = foldm.groupby("DAS")[["PCC", "MSE"]].std()
    for m in ("PCC", "MSE"):
        tab[f"{s}_{m}"]         = [pooled.loc[t, (m, "mean")] for t in tab.DAS]
        tab[f"{s}_{m}_sd"]      = [pooled.loc[t, (m, "std")]  for t in tab.DAS]
        tab[f"{s}_{m}_sd_fold"] = [foldsd.loc[t, m]           for t in tab.DAS]
tab.to_csv(os.path.join(OUT, "maize_rgb_per_timepoint_v2.csv"), index=False)
print("per_timepoint saved:", os.path.join(OUT, "maize_rgb_per_timepoint_v2.csv"), tab.shape)
print()
for s in SER:
    print("%-26s PCC=%.4f (pooledSD=%.4f foldSD=%.4f)  MSE=%.5f (%.5f/%.5f)" % (
        s, tab[s+"_PCC"].mean(), tab[s+"_PCC_sd"].mean(), tab[s+"_PCC_sd_fold"].mean(),
        tab[s+"_MSE"].mean(), tab[s+"_MSE_sd"].mean(), tab[s+"_MSE_sd_fold"].mean()))
