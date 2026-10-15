#!/usr/bin/env python3
"""
Build VanRaden GRM from MAGIC SNP data — compare both approaches:
  A) Compute full 552×552 then subset → 330×330  (matches author: p from 552)
  B) Subset to 330 then compute 330×330           (standard: p from 330)

Input:  MAGIC.SNP.70k.impute.maf.vcf.txt (552 genotypes × 70846 markers, 0/1/2)
Output: GRM_330x330_methodA.csv, GRM_330x330_methodB.csv, comparison stats
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r'D:\pythonproject\knode\data\maize_magic')
OUT_DIR  = Path(r'D:\pythonproject\knode\data\maize_magic')
MAF_THRESH = 0.05

def vanraden_grm(M):
    """
    VanRaden (2008) Method 1.
    M: n×m genotype matrix (0/1/2)
    Returns: G (n×n), p (m,), maf (m,), denominator (scalar)
    """
    p = M.mean(axis=0) / 2.0
    maf = np.minimum(p, 1.0 - p)
    keep = maf >= MAF_THRESH
    M_qc = M[:, keep]
    p_qc = p[keep]
    maf_qc = maf[keep]

    Z = M_qc - 2.0 * p_qc
    denom = 2.0 * np.sum(p_qc * (1.0 - p_qc))
    G = (Z @ Z.T) / denom
    return G, p_qc, maf_qc, denom, keep


# ============================================================
# 1. Load
# ============================================================
print('=' * 60)
print('Loading SNP matrix...')
snp_df = pd.read_csv(DATA_DIR / 'MAGIC.SNP.70k.impute.maf.vcf.txt', sep=' ', index_col=0)
M_all = snp_df.values.astype(np.float32)
all_ids = snp_df.index.tolist()
print(f'  Full: {M_all.shape[0]} genotypes × {M_all.shape[1]} markers')

# Phenotype IDs (330)
pheno = pd.read_csv(DATA_DIR / 'maize_trait_BLUP_50_NArmv_traits_afterClustering.csv')
pheno_ids = sorted(set(pheno['bio_ID'].astype(str)))
common_ids = sorted(set(all_ids) & set(pheno_ids))
print(f'  Phenotype genotypes: {len(pheno_ids)}')
print(f'  Common: {len(common_ids)}')

# ============================================================
# 2. Method A: compute full 552×552 then subset
# ============================================================
print('\n' + '=' * 60)
print('Method A: Compute 552×552, then subset to 330')
print('=' * 60)
G_full, p_a, maf_a, denom_a, keep_a = vanraden_grm(M_all)
print(f'  Markers after MAF filter: {len(p_a)}')
print(f'  Denominator: {denom_a:.2f}')
print(f'  G_full diag mean: {np.diag(G_full).mean():.4f}')

grm_a = pd.DataFrame(G_full, index=all_ids, columns=all_ids)
grm_a_sub = grm_a.loc[common_ids, common_ids]
print(f'  Subset GRM: {grm_a_sub.shape}')
print(f'  Subset diag mean: {np.diag(grm_a_sub.values).mean():.4f}')

grm_a_sub.to_csv(OUT_DIR / 'GRM_330x330_methodA_from552.csv')
print('  Saved: GRM_330x330_methodA_from552.csv')

# ============================================================
# 3. Method B: subset to 330 first, then compute
# ============================================================
print('\n' + '=' * 60)
print('Method B: Subset to 330, then compute')
print('=' * 60)
M_sub = snp_df.loc[common_ids].values.astype(np.float32)
G_b, p_b, maf_b, denom_b, keep_b = vanraden_grm(M_sub)
print(f'  Markers after MAF filter: {len(p_b)}')
print(f'  Denominator: {denom_b:.2f}')
print(f'  G_sub diag mean: {np.diag(G_b).mean():.4f}')

grm_b = pd.DataFrame(G_b, index=common_ids, columns=common_ids)
grm_b.to_csv(OUT_DIR / 'GRM_330x330_methodB_from330.csv')
print('  Saved: GRM_330x330_methodB_from330.csv')

# ============================================================
# 4. Comparison
# ============================================================
print('\n' + '=' * 60)
print('Comparison: A (from 552) vs B (from 330) vs Old GCTA')
print('=' * 60)

va = grm_a_sub.values.flatten()
vb = grm_b.values.flatten()
corr_ab = np.corrcoef(va, vb)[0, 1]
mae_ab = np.mean(np.abs(grm_a_sub.values - grm_b.values))
print(f'  Corr(A, B): {corr_ab:.6f}')
print(f'  MAE(A, B):  {mae_ab:.6f}')
print(f'  Max diff:   {np.max(np.abs(grm_a_sub.values - grm_b.values)):.6f}')
print(f'  A diag:     {np.diag(grm_a_sub.values).mean():.4f} ± {np.diag(grm_a_sub.values).std():.4f}')
print(f'  B diag:     {np.diag(grm_b.values).mean():.4f} ± {np.diag(grm_b.values).std():.4f}')

# Compare with old GCTA GRM
old_path = DATA_DIR / 'GRM_330x330_gcta.csv'
if old_path.exists():
    old_grm = pd.read_csv(old_path, header=None)
    print(f'\n  Old GRM ({old_grm.shape[0]}×{old_grm.shape[1]}):')
    if old_grm.shape[0] == len(common_ids):
        vo = old_grm.values.flatten()
        corr_ao = np.corrcoef(va, vo)[0, 1]
        corr_bo = np.corrcoef(vb, vo)[0, 1]
        mae_ao = np.mean(np.abs(grm_a_sub.values - old_grm.values))
        mae_bo = np.mean(np.abs(grm_b.values - old_grm.values))
        print(f'  Corr(old, A): {corr_ao:.6f}  MAE: {mae_ao:.6f}')
        print(f'  Corr(old, B): {corr_bo:.6f}  MAE: {mae_bo:.6f}')

# ============================================================
# 5. Eigendecomposition check (positive semi-definite)
# ============================================================
print('\n' + '=' * 60)
print('PSD check (eigenvalues)')
print('=' * 60)
for name, G in [('A (from 552)', grm_a_sub.values), ('B (from 330)', grm_b.values)]:
    eigvals = np.linalg.eigvalsh(G)
    n_neg = (eigvals < -1e-8).sum()
    print(f'  {name}: min eig={eigvals.min():.6f}, max={eigvals.max():.4f}, neg={n_neg}')

print('\nDone.')
