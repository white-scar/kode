# -*- coding: utf-8 -*-
"""_author_rice_Xnorm.py -- 为作者 rice 后处理生成每折的 val_X_norm.csv (训练折 scaler 归一化, trait-major 展平)
行 = trait x time (trait-major: 每个性状的 T 个时间点连续排列), 列 = 验证集基因型 ID
"""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/root/autodl-tmp/qkn")
COMP = ROOT / "outputs/reproduce/author_tp1_rice/components"
pheno = pd.read_csv(ROOT / "data/rice_salinity/rice_indica_control_traits.csv")
pheno = pheno[pheno["DAS"] >= 30].copy()
pheno["bio_ID"] = pheno["bio_ID"].astype(str)
cv = pd.read_csv(ROOT / "data/rice_salinity/rice_cv_10fold.csv", index_col=0)
cv.index = cv.index.astype(str)
meta = ["Unnamed: 0", "bio_ID", "accession", "DAS"]
trait_cols = [c for c in pheno.columns if c not in meta and not c.startswith("Unnamed") and c != ""]
times = sorted(pheno["DAS"].unique()); T = len(times); P = len(trait_cols)
gids = sorted(set(pheno["bio_ID"]) & set(cv.index))
print("P=%d T=%d N=%d" % (P, T, len(gids)), flush=True)
for k in range(1, 11):
    fc = cv.loc[gids].iloc[:, k - 1].values.astype(int)
    trn = [g for g, f in zip(gids, fc) if f != 6]; val = [g for g, f in zip(gids, fc) if f == 6]
    dtr = pheno[pheno.bio_ID.isin(trn)]
    mn = dtr[trait_cols].min().values; mx = dtr[trait_cols].max().values
    rg = np.maximum(mx - mn, 1e-8)
    M = np.zeros((P * T, len(val)))
    for gi, g in enumerate(val):
        sub = pheno[pheno.bio_ID == g].sort_values("DAS")
        assert len(sub) == T
        M[:, gi] = ((sub[trait_cols].values - mn) / rg).T.ravel()      # (P,T) trait-major
    pd.DataFrame(M, columns=val).to_csv(COMP / ("iter%02d_val_X_norm.csv" % k))
    print("  fold %d: %d x %d (P*T x n_val)" % (k, M.shape[0], M.shape[1]), flush=True)
print("XNORM DONE")