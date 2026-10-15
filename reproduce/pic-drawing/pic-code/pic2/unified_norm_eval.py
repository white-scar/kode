# -*- coding: utf-8 -*-
"""unified_norm_eval.py — 统一归一化口径下, 作者三个变体 与 Kode 的逐点 PCC/MSE

口径(三条):
  1) scaler 只在训练折(folds1-5)拟合: 每性状 min/max 取训练折内(基因型 x 时间)极值
  2) 训练/验证/验证真值共用同一套 scaler -> MSE 天然同空间
  3) 作者 DMD 用 scale='none' (禁用其内部二次 min-max)

SD 定义 (与 pic2_maize.R / make_figE4_data.py 一致):
  pooled SD = 对 (trait x fold/iteration) 池化后求 std   <-- 主用
  另存 _sd_fold = 先折内对 trait 平均、再对折求 std       <-- 备用

作者变体:
  tp1  = tp1+MegaLMM+dynamicGP  -> inner-5 集成 (author_tp1_maize_innerens/components)
  mega = MegaLMM+dynamicGP      -> 单次拟合 (dynamicgp_megalmm/snapshot_metrics_long.csv)
  rr   = rrBLUP+dynamicGP       -> 单次拟合 (同上)

输出:
  outputs/kode_result/maize/unified_norm_author_vs_kode_per_timepoint.csv
  outputs/kode_result/maize/unified_norm_author_vs_kode_per_foldset.csv  (逐 trait x fold 长表)
"""
import numpy as np, pandas as pd, os
BASE = r"D:\pythonproject\knode"
HERE = os.path.dirname(os.path.abspath(__file__))   # write unified tables next to the fig2 script
OUT  = os.path.join(BASE, "outputs", "kode_result", "maize")   # Kode per-trait source (read-only)
DATA = os.path.join(BASE, "data", "maize_magic")

DAS_ALL = [15,16,17,18,19,22,23,24,25,26,29,30,31,32,33,36,37,38,39,40,43,44,45,46,47]
dasmap  = {f"t{i+1}": d for i, d in enumerate(DAS_ALL)}

# ---------------- 数据 ----------------
pheno = pd.read_csv(os.path.join(DATA, "maize_trait_BLUP_50_NArmv_traits_afterClustering.csv"))
cv    = pd.read_csv(os.path.join(DATA, "cross_validation_folds_w_validation_set_maize.csv"), index_col=0)
meta  = ["bio_ID", "accession", "DAS"]
traits = [c for c in pheno.columns if c not in meta and not c.startswith("Unnamed") and c != ""]
gids = sorted(set(pheno["bio_ID"].astype(str)) & set(cv.index))
times = DAS_ALL
Y = np.full((len(gids), len(times), len(traits)), np.nan)
for pi, tr in enumerate(traits):
    s = pheno[["bio_ID","DAS",tr]].copy(); s["bio_ID"] = s["bio_ID"].astype(str)
    Y[:, :, pi] = s.pivot(index="bio_ID", columns="DAS", values=tr).loc[gids].values

# ---------------- 作者 tp1: inner-5 集成, 逐折 rollout (逐 trait 记录) ----------------
CVdir = os.path.join(BASE, "outputs", "reproduce", "author_tp1_maize_innerens", "components")
rows = []
for k in range(1, 11):
    fc = cv.loc[gids].iloc[:, k-1].values.astype(int)
    tr = np.where((fc>=1)&(fc<=5))[0]; te = np.where(fc==6)[0]
    tr_min = np.nanmin(Y[tr], axis=(0,1)); tr_rng = np.maximum(np.nanmax(Y[tr],axis=(0,1))-tr_min, 1e-8)
    Yte = ((Y[te]-tr_min)/tr_rng)
    Rp = pd.read_csv(os.path.join(CVdir, f"iter{k:02d}_val_R_enspred.csv"), index_col=0)
    Pp = pd.read_csv(os.path.join(CVdir, f"iter{k:02d}_val_phi_enspred.csv"), index_col=0)
    te_ids = [gids[i] for i in te]
    Rp = Rp.loc[te_ids]; Pp = Pp.loc[te_ids]
    P = len(traits)
    A = np.zeros((len(te_ids), P, P))
    for i in range(len(te_ids)):
        Rm = Rp.iloc[i].values.reshape(2,2)
        Pm = Pp.iloc[i].values.reshape(P,2)
        A[i] = Pm @ Rm @ np.linalg.pinv(Pm)
    pred = np.zeros_like(Yte); pred[:,0,:] = Yte[:,0,:]
    for t in range(1, len(times)):
        pred[:,t,:] = np.einsum('bp,bqp->bq', pred[:,t-1,:], A)
    pit = np.zeros_like(Yte); pit[:,0,:] = Yte[:,0,:]
    for t in range(1, len(times)):
        pit[:,t,:] = np.einsum('bp,bqp->bq', Yte[:,t-1,:], A)
    for t in range(1, len(times)):
        for tag, P_ in [("rec", pred), ("iter", pit)]:
            for p in range(P):
                a = P_[:,t,p]; b = Yte[:,t,p]
                if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                    r = np.corrcoef(a, b)[0,1]
                else:
                    r = np.nan
                mse_ = float(np.mean((a-b)**2))
                vy_ = float(np.var(b))
                rows.append(dict(Series=f"author_tp1_{tag}", Fold=k, DAS=int(times[t]),
                                 trait=traits[p], PCC=float(r), MSE=mse_,
                                 R2=(1-mse_/vy_) if vy_ > 0 else np.nan,
                                 NMSE=(mse_/vy_) if vy_ > 0 else np.nan))
AUT = pd.DataFrame(rows)

# ---------------- 作者 mega / rr: 单次拟合 (逐 trait 长表) ----------------
# 2026-09-27: inner-5 split scripts -> mega and rr now live in separate dirs
long = pd.concat([
    pd.read_csv(os.path.join(BASE, "outputs", "dynamicgp_megalmm_innerens", "snapshot_metrics_long.csv")),
    pd.read_csv(os.path.join(BASE, "outputs", "dynamicgp_rr_innerens",        "snapshot_metrics_long.csv")),
], ignore_index=True)
long["DAS"] = long["Time"].map(dasmap)
long = long[long["DAS"].notna() & (long["DAS"] != 15)]
SF = long[long.Method.isin(["mega_rec","mega_iter","rr_rec","rr_iter"])].copy()
SF["Prot"] = np.where(SF.Method.str.endswith("_iter"), "iter", "rec")
SF["Var"]  = np.where(SF.Method.str.startswith("mega"), "mega", "rr")
SF["Series"] = "author_" + SF.Var + "_" + SF.Prot
SFR = SF.rename(columns={"Iteration":"Fold"})[["Series","Fold","DAS","Trait","PCC","MSE","R2","NMSE"]].rename(columns={"Trait":"trait"})

# ---------------- 本方法: 逐 trait x fold ----------------
kd = []
for mode, prot in [("ode_hyper","rec"), ("iter_hyper","iter")]:      # 2026-09-14 统一为无 skip 档
    d = pd.read_csv(os.path.join(OUT, f"knode_tp1_{mode}_per_trait.csv"))
    d = d[(d.split=="test") & (d.timepoint!=15)]
    kd.append(pd.DataFrame(dict(Series=f"Kode_{prot}", Fold=d.foldset.values,
                                DAS=d.timepoint.values, trait=d.trait.values,
                                PCC=d.pcc.values, MSE=d.MSE_norm.values,
                                R2=d.R2.values, NMSE=(1.0 - d.R2.values))))
KD = pd.concat(kd, ignore_index=True)

# ---------------- 汇总 ----------------
fold = pd.concat([AUT, SFR, KD], ignore_index=True)
fold = fold[["Series","Fold","DAS","trait","PCC","MSE","R2","NMSE"]].sort_values(["Series","DAS"])
fold.to_csv(os.path.join(HERE, "unified_norm_author_vs_kode_per_foldset.csv"), index=False)

SERIES_ORDER = ["author_tp1_rec","author_tp1_iter","author_mega_rec","author_mega_iter",
                "author_rr_rec","author_rr_iter","Kode_rec","Kode_iter"]
tab = pd.DataFrame({"DAS": [t for t in times if t != 15]})
for s in SERIES_ORDER:
    sub = fold[fold.Series==s]
    pooled = sub.groupby("DAS")[["PCC","MSE","R2","NMSE"]].agg(["mean","std"])            # pooled over trait x fold
    foldm  = sub.groupby(["DAS","Fold"])[["PCC","MSE","R2","NMSE"]].mean()                 # 折内先对 trait 平均
    foldsd = foldm.groupby("DAS")[["PCC","MSE","R2","NMSE"]].std()
    for metric in ("PCC","MSE","R2","NMSE"):
        tab[f"{s}_{metric}"]         = [pooled.loc[t,(metric,"mean")] for t in tab.DAS]
        tab[f"{s}_{metric}_sd"]      = [pooled.loc[t,(metric,"std")]  for t in tab.DAS]
        tab[f"{s}_{metric}_sd_fold"] = [foldsd.loc[t,  metric]        for t in tab.DAS]

out_csv = os.path.join(HERE, "unified_norm_author_vs_kode_per_timepoint.csv")
tab.to_csv(out_csv, index=False)

print("=== 24 点均值 (pooled SD / fold SD) ===")
for s in SERIES_ORDER:
    print("%-16s PCC=%.4f (pooledSD=%.4f foldSD=%.4f)  MSE=%.5f (%.5f / %.5f)" % (
        s, tab[s+"_PCC"].mean(), tab[s+"_PCC_sd"].mean(), tab[s+"_PCC_sd_fold"].mean(),
        tab[s+"_MSE"].mean(), tab[s+"_MSE_sd"].mean(), tab[s+"_MSE_sd_fold"].mean()))
print()
print("columns:", len(tab.columns), " folded rows:", len(fold))
print("saved ->", out_csv)
