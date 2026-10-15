# -*- coding: utf-8 -*-
import zipfile, io, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(errors='replace')
ours = pd.read_csv('data/maize_drought/SNP/geno.numerical.csv', index_col=0); ours.index = ours.index.astype(str)
z = zipfile.ZipFile(r'C:\Users\27342\Downloads\ICIM_Infiles.zip')
f = pd.read_csv(io.BytesIO(z.read('ICIM_Infiles/RawFiles/Weibull/YC2D_ICIMap_Weibullallelecalls_SNPbinner.csv')))
tax = [c for c in f.columns if c not in ('marker','chromosome','position.bp.')]
f = f.set_index('marker')
com_t = [t for t in tax if t in ours.index]; com_m = [m for m in f.index if m in ours.columns]
A = f.loc[com_m, com_t].astype(float); B = ours.loc[com_t, com_m].T.astype(float)
hi, lo = [], []
for m in com_m:
    a, b = A.loc[m].values, B.loc[m].values
    ok = np.isfinite(a) & np.isfinite(b)
    m2 = ok & (a != 1)
    if m2.sum() < 20 or np.std(a[m2]) < 1e-9 or np.std(b[m2]) < 1e-9: continue
    r = np.corrcoef(a[m2], b[m2])[0, 1]
    (hi if r > 0.9 else lo).append((m, r))
print('?????:', hi[:3])
print('?????:', lo[:3])
print()
for tag, lst in [('???', hi[:2]), ('???', lo[:2])]:
    for m, r in lst:
        a = A.loc[m].values; b = B.loc[m].values
        ca = np.bincount(np.nan_to_num(a).astype(int), minlength=3)[:3]
        cb = np.bincount(np.nan_to_num(b).astype(int), minlength=3)[:3]
        print('=== %s  %-14s corr=%.3f' % (tag, m, r))
        print('    ?? 0/1/2 ??: %s   ?? 0/1/2 ??: %s' % (ca, cb))
        print('    ? 12 ??? (Taxa: ?? / ??):')
        print('     ', [(t, a[i], b[i]) for i, t in enumerate(com_t[:12])])
