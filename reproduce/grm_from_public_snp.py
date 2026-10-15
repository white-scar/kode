#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""? Anderson 2020 ??? ICIM ???(??? binned)? GRM, ???? geno_GRM ??."""
import zipfile, io, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(errors='replace')

ours = pd.read_csv('data/maize_drought/SNP/geno.numerical.csv', index_col=0); ours.index = ours.index.astype(str)
grm_ours = pd.read_csv('data/maize_drought/geno_GRM.csv', index_col=0); grm_ours.index = grm_ours.index.astype(str)

def vanraden(X, maf=0.05):
    # X: (n, m) 0/1/2, ???????
    p = X.mean(0) / 2.0
    keep = (p > maf) & (p < 1 - maf)
    X = X[:, keep]; p = p[keep]
    Z = X - 2 * p
    denom = 2.0 * np.sum(p * (1 - p))
    return (Z @ Z.T) / denom, keep.sum()

# ---- 0) ????????? GRM, ???? ----
Xo = ours.values.astype(float)
G_o, m_o = vanraden(Xo)
common0 = sorted(set(G_o.shape) and set(ours.index))
G_o_df = pd.DataFrame(G_o, index=ours.index, columns=ours.index)
cmp_idx = sorted(set(ours.index) & set(grm_ours.index))
a = G_o_df.loc[cmp_idx, cmp_idx].values; b = grm_ours.loc[cmp_idx, cmp_idx].values
off = ~np.eye(len(cmp_idx), dtype=bool)
print('=== 0) ??: ???? geno.numerical ?? GRM vs ?? geno_GRM ===')
print('   ???(MAF>0.05 ?? %d/%d) | ?? %d' % (m_o, ours.shape[1], len(cmp_idx)))
print('   ??(???)?? %.4f vs %.4f | ??? max|?|=%.4f mean|?|=%.4f ??=%.4f' % (
    np.diag(a).mean(), np.diag(b).mean(), np.abs(a-b)[off].max(), np.abs(a-b)[off].mean(),
    np.corrcoef(a[off], b[off])[0,1]))
print()

# ---- 1) ????????? binned ???, ??????? ----
z = zipfile.ZipFile(r'C:\Users\27342\Downloads\ICIM_Infiles.zip')
files = sorted(n for n in z.namelist() if n.endswith('allelecalls_SNPbinner.csv'))
blocks = []
for n in files:
    f = pd.read_csv(io.BytesIO(z.read(n)))
    taxa = [c for c in f.columns if c not in ('marker', 'chromosome', 'position.bp.')]
    f = f.set_index('marker')[taxa].T
    f.index = f.index.astype(str)
    f.columns = f.columns.astype(str)
    blocks.append(f)
    print('   %-30s %s' % (n.split('/')[-1][:30], f.shape))
alltaxa = sorted(set().union(*[set(b.index) for b in blocks]))
allmark = sorted(set().union(*[set(b.columns) for b in blocks]))
print()
print('=== 1) ????????: %d ?? ? %d ??' % (len(alltaxa), len(allmark)))
M = pd.DataFrame(np.nan, index=alltaxa, columns=allmark)
for b in blocks:
    M.loc[b.index, b.columns] = b.values
print('   ??? %.1f%% (???"????????????" -> ????)' % (100 * M.isna().mean().mean()))
X = M.values.astype(float)
colmean = np.nanmean(X, axis=0)
idx = np.where(np.isnan(X))
X[idx] = np.take(colmean, idx[1])
G_t, m_t = vanraden(X)
G_t_df = pd.DataFrame(G_t, index=alltaxa, columns=alltaxa)
print('   MAF>0.05 ?? %d ?? | GRM %s' % (m_t, G_t.shape))
print()

# ---- 2) ?? GRM ?? (???? 485) ----
com = sorted(set(alltaxa) & set(ours.index))
A = G_t_df.loc[com, com].values
B = G_o_df.loc[com, com].values
off = ~np.eye(len(com), dtype=bool)
print('=== 2) ??? GRM vs ??? GRM (?? %d ???) ===' % len(com))
print('   ???: ?? %.4f | max|?| %.4f | mean|?| %.4f' % (
    np.corrcoef(A[off], B[off])[0,1], np.abs(A-B)[off].max(), np.abs(A-B)[off].mean()))
wa, vb = np.linalg.eigvalsh(A)[::-1], np.linalg.eigvalsh(B)[::-1]
print('   ???: ?5 ?? %s' % np.round(wa[:5], 1), '| ?? %s' % np.round(vb[:5], 1))
np.save('_Gt_public.npy', A); np.save('_Go_ours.npy', B)
open('_common_ids.txt', 'w').write('\n'.join(com))
print('   (?? _Gt_public.npy / _Go_ours.npy / _common_ids.txt)')
