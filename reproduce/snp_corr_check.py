# -*- coding: utf-8 -*-
import zipfile, io, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(errors='replace')
ours = pd.read_csv('data/maize_drought/SNP/geno.numerical.csv', index_col=0); ours.index = ours.index.astype(str)
z = zipfile.ZipFile(r'C:\Users\27342\Downloads\ICIM_Infiles.zip')
for pop in ['YC2D', 'YC2I', 'YC1D', 'YC3D']:
    f = pd.read_csv(io.BytesIO(z.read('ICIM_Infiles/RawFiles/Weibull/%s_ICIMap_Weibullallelecalls_SNPbinner.csv' % pop)))
    tax = [c for c in f.columns if c not in ('marker','chromosome','position.bp.')]
    f = f.set_index('marker')
    com_t = [t for t in tax if t in ours.index]; com_m = [m for m in f.index if m in ours.columns]
    A = f.loc[com_m, com_t].astype(float).values
    B = ours.loc[com_t, com_m].T.astype(float).values
    cors, freq_ok = [], 0
    for i in range(len(com_m)):
        a, b = A[i], B[i]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 20 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
            continue
        # ??????????, ????
        m2 = ok & (a != 1)
        if m2.sum() < 20 or np.std(a[m2]) < 1e-9 or np.std(b[m2]) < 1e-9:
            continue
        cors.append(np.corrcoef(a[m2], b[m2])[0, 1])
        pa, pb = a[m2].mean() / 2, b[m2].mean() / 2
        freq_ok += abs(pa - pb) < 0.05
    c = np.array(cors)
    print('=== %s: ? %d ?? | ???(?? vs ??)??: ?? %.3f ?? %.3f | ??>0.9 ?? %.1f%% | ???????? %.1f%%' % (
        pop, len(c), c.mean(), np.median(c), 100*(c > 0.9).mean(), 100*freq_ok/max(len(c),1)))
