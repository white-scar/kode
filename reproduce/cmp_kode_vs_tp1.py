#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kode vs DynamicGP-MegaLMM+TP1 ???? ? rec/iter ????."""
import pandas as pd, numpy as np, os, sys
sys.stdout.reconfigure(errors='replace')
SRC = {
 '?? MAGIC': 'reproduce/pic-code/pic2/unified_norm_author_vs_kode_per_timepoint.csv',
 '???':     'reproduce/pic-code/pic2/arab-E1/unified_norm_author_vs_kode_per_timepoint.csv',
 '??':       'reproduce/pic-code/pic2/rice-E2/unified_norm_author_vs_kode_per_timepoint.csv',
}
def col(df, s, m):
    c = '%s_%s' % (s, m)
    return df[c].values if c in df.columns else None
print('%-10s %-9s %9s %9s %9s | %9s %9s %9s | %s' % (
    '???','??','Kode_PCC','tp1_PCC','?PCC','Kode_MSE','tp1_MSE','MSE??','???(?)'))
for name, p in SRC.items():
    if not os.path.exists(p):
        print('  (?)', p); continue
    d = pd.read_csv(p)
    for prot in ['rec','iter']:
        kn, kt = 'Kode_%s' % prot, 'author_tp1_%s' % prot
        pk, pt = col(d, kn, 'PCC'), col(d, kt, 'PCC')
        mk, mt = col(d, kn, 'MSE'), col(d, kt, 'MSE')
        if pk is None or pt is None: continue
        dp = pk - pt
        best = int(np.argmax(dp))
        das = d['DAS'].values   # ?? 24 ?; rice/arab ??
        print('%-10s %-9s %9.4f %9.4f %+9.4f | %9.5f %9.5f %8.1f%% | DAS %s (?=%+.3f)' % (
            name, prot, pk.mean(), pt.mean(), dp.mean(), mk.mean(), mt.mean(),
            100*(1-mk.mean()/mt.mean()), das[best], dp[best]))
    print()
print()
print('=== ???????Kode vs tp1 ? PCC?===')
for name, p in SRC.items():
    d = pd.read_csv(p)
    for prot in ['rec','iter']:
        kn, kt = 'Kode_%s' % prot, 'author_tp1_%s' % prot
        pk, pt = col(d, kn, 'PCC'), col(d, kt, 'PCC')
        if pk is None: continue
        print('  --- %s / %s' % (name, prot))
        print('      DAS   :', ' '.join('%6d' % x for x in d.DAS.values))
        print('      Kode  :', ' '.join('%6.3f' % x for x in pk))
        print('      tp1   :', ' '.join('%6.3f' % x for x in pt))
        print('      ?     :', ' '.join('%+6.3f' % x for x in (pk-pt)))
