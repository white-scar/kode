# -*- coding: utf-8 -*-
import zipfile, io, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(errors='replace')
ours = pd.read_csv('data/maize_drought/SNP/geno.numerical.csv', index_col=0); ours.index = ours.index.astype(str)
z = zipfile.ZipFile(r'C:\Users\27342\Downloads\ICIM_Infiles.zip')
print('%-6s %7s %10s %10s %10s %12s' % ('pop','???','??(corr>0.9)','??(corr<-0.9)','??','??????'))
for pop in ['YC1D','YC2D','YC3D']:
    f = pd.read_csv(io.BytesIO(z.read('ICIM_Infiles/RawFiles/Weibull/%s_ICIMap_Weibullallelecalls_SNPbinner.csv' % pop)))
    tax = [c for c in f.columns if c not in ('marker','chromosome','position.bp.')]
    f = f.set_index('marker')
    com_t = [t for t in tax if t in ours.index]; com_m = [m for m in f.index if m in ours.columns]
    A = f.loc[com_m, com_t].astype(float).values
    B = ours.loc[com_t, com_m].T.astype(float).values
    pos = neg = oth = 0; al = []; tot_ok = tot_n = 0
    for i in range(len(com_m)):
        a, b = A[i], B[i]
        m2 = np.isfinite(a) & np.isfinite(b) & (a != 1)
        if m2.sum() < 20: continue
        x, y = a[m2], b[m2]
        if np.std(x) < 1e-9 or np.std(y) < 1e-9: continue
        r = np.corrcoef(x, y)[0, 1]
        if r > 0.9: pos += 1; ai, bi = x, y
        elif r < -0.9: neg += 1; ai, bi = x, 2 - y
        else: oth += 1; continue
        tot_ok += (ai == bi).sum(); tot_n += len(ai)
    print('%-6s %7d %10d %10d %10d %11.2f%%' % (pop, pos+neg+oth, pos, neg, oth, 100*tot_ok/max(tot_n,1)))
print()
print('?: "??"= ? corr<-0.9 ???? 2-x ??; ???????????')
