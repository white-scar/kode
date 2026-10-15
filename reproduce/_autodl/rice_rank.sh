#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
/root/miniconda3/bin/python - <<'PY'
import glob, pandas as pd, numpy as np
# rank ?? (????): ?? fold ?? csv, ? cond x rank ? pcc_mean
for ds in ['rice','uav']:
    fs = sorted(glob.glob('/root/autodl-tmp/qkn/outputs/reproduce/knode_rank_genetics_value_%s_f*.csv' % ds))
    if not fs:
        print('=== %s: ? per-fold ??' % ds); continue
    d = pd.concat([pd.read_csv(f) for f in fs])
    print('=== %s rank ?? (%d ?) ? ??? PCC ===' % (ds, d.fold.nunique() if 'fold' in d.columns else len(fs)))
    piv = d.pivot_table(index='rank', columns='cond', values='pcc_mean', aggfunc='mean')
    cols = [c for c in ['full','tp1only','grmonly','const','noise'] if c in piv.columns]
    print(piv[cols].round(4).to_string())
    if 'full' in piv.columns and 'tp1only' in piv.columns:
        print('   full - tp1only:', (piv['full']-piv['tp1only']).round(4).to_dict())
    print()
PY
