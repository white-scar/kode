# -*- coding: utf-8 -*-
import zipfile, io, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(errors='replace')
ours = pd.read_csv('data/maize_drought/SNP/geno.numerical.csv', index_col=0); ours.index = ours.index.astype(str)
z = zipfile.ZipFile(r'C:\Users\27342\Downloads\ICIM_Infiles.zip')
print('=== ????????????? vs ??????===')
print('%-6s %8s %10s %10s %12s %12s' % ('pop', '???', '????', '??(??????)', '???', '???(???)'))
tot = [0, 0, 0]
for n in sorted(x for x in z.namelist() if x.endswith('allelecalls_SNPbinner.csv')):
    f = pd.read_csv(io.BytesIO(z.read(n)))
    tax = [c for c in f.columns if c not in ('marker','chromosome','position.bp.')]
    f = f.set_index('marker')
    tag = n.split('/')[-1].split('_')[0]
    com_t = [t for t in tax if t in ours.index]
    com_m = [m for m in f.index if m in ours.columns]
    sub_t = f.loc[com_m, com_t].astype(float)
    ref = ours.loc[com_t, com_m].T.astype(float)          # ???
    same = (sub_t.values == ref.values)
    het = (sub_t.values == 1)
    diff = ~same & ~het
    n_cell = same.size
    print('%-6s %8d %10d %10d %12d %11.2f%%' % (tag, n_cell, same.sum(), het.sum(), diff.sum(),
          100 * same.sum() / max(n_cell - het.sum(), 1)))
    tot[0] += n_cell; tot[1] += same.sum(); tot[2] += het.sum()
    if tag == 'YC2D':
        idx = np.argwhere(diff)
        print('       ?????(?5): %s' % [(f.index[i], f.columns[j], sub_t.values[i,j], ref.values[i,j]) for i,j in idx[:5]])
print()
print('=== ??: ?? %d | ???? %d (%.1f%%) | ?????? %d (%.1f%%) | ???? %d (%.2f%%) ===' % (
    tot[0], tot[1], 100*tot[1]/tot[0], tot[2], 100*tot[2]/tot[0], tot[0]-tot[1]-tot[2],
    100*(tot[0]-tot[1]-tot[2])/tot[0]))
