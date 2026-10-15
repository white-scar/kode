#!/bin/bash
/root/miniconda3/bin/python - <<'PY'
import glob, pandas as pd
for ds in ['rice','uav']:
    fs = sorted(glob.glob('/root/autodl-tmp/qkn/outputs/reproduce/knode_rank_genetics_value_%s_f*.csv' % ds))
    if not fs:
        print('=== %s ???' % ds); continue
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    print('=== %s rank ??: %d ?, ?=%s' % (ds, len(fs), list(d.columns)[:4]))
    piv = d.pivot_table(index='rank', columns='cond', values='pcc', aggfunc='mean')
    cols = [c for c in ['full','tp1only','grmonly','const','noise'] if c in piv.columns]
    print(piv[cols].round(4).to_string())
    if 'full' in piv.columns and 'tp1only' in piv.columns:
        diff = (piv['full'] - piv['tp1only']).round(4)
        print('   full - tp1only:', diff.to_dict())
    print()
PY
