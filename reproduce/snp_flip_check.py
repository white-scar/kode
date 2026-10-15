# -*- coding: utf-8 -*-
import zipfile, io, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(errors='replace')
ours = pd.read_csv('data/maize_drought/SNP/geno.numerical.csv', index_col=0); ours.index = ours.index.astype(str)
z = zipfile.ZipFile(r'C:\Users\27342\Downloads\ICIM_Infiles.zip')
f = pd.read_csv(io.BytesIO(z.read('ICIM_Infiles/RawFiles/Weibull/YC2D_ICIMap_Weibullallelecalls_SNPbinner.csv')))
tax = [c for c in f.columns if c not in ('marker','chromosome','position.bp.')]
f = f.set_index('marker')
com_t = [t for t in tax if t in ours.index]; com_m = [m for m in f.index if m in ours.columns]
A = f.loc[com_m, com_t].astype(float).values        # theirs
B = ours.loc[com_t, com_m].T.astype(float).values   # ours
mask = (A != 1) & np.isfinite(A) & np.isfinite(B)   # ???????
agree = (A == B)
print('=== YC2D: ????????????????===')
per = agree[mask].reshape(len(com_m), -1) if False else None
rows = []
for i in range(len(com_m)):
    m = mask[i]
    if m.sum() < 20: continue
    rows.append(agree[i][m].mean())
r = np.array(rows)
print('   ??? %d | ???: ?? %.3f ?? %.3f' % (len(r), r.mean(), np.median(r)))
import collections
bins = collections.Counter((r > 0.95, (r < 0.05), ((r > 0.4) & (r < 0.6))))
print('   ~100%% ??: %d ? | ~0%% ??(=????): %d ? | ??(0.4-0.6): %d ?' % (bins[(True,False,False)], bins[(False,True,False)], bins[(False,False,True)]))
print()
print('=== ??"????"?????(? 2-?), ????? ===')
flip = r < 0.5
B2 = B.copy()
B2[:, flip] = 2 - B2[:, flip]
mask2 = mask
print('   ????????: %.2f%%' % (100 * (A[mask2] == B2[mask2]).mean()))
print()
print('=== ??????? ===')
for k in range(3):
    i = k
    m = mask[i]
    if m.sum() < 20: continue
    print('   %-14s chr%-3s ??? %.2f | ?? 0/2 ?? %s | ?? 0/2 ?? %s' % (
        com_m[i], f.loc[com_m[i], 'chromosome'], r[i],
        np.bincount(A[i][m].astype(int), minlength=3)[::2], np.bincount(B[i][m].astype(int), minlength=3)[::2]))
