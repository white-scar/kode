#!/bin/bash
cd /root/autodl-tmp/qkn/outputs/kode_result/arab
/root/miniconda3/bin/python - <<'PY'
import glob, os, pandas as pd
rows = []
for f in sorted(glob.glob('knode_arab_tp1skip_all_summary_hp_*.csv')):
    cfg = os.path.basename(f).replace('knode_arab_tp1skip_all_summary_hp_','').replace('.csv','')
    d = pd.read_csv(f)
    for _, r in d[d.split=='test'].iterrows():
        rows.append(dict(config=cfg, mode=r['mode'], pcc=float(r['pcc']),
                         mse=float(r['MSE_norm']), r2m=float(r['R2_median']), rlong=float(r['r_long'])))
t = pd.DataFrame(rows)
print('???:', t.config.nunique(), '| mode ??:', sorted(t['mode'].unique()))
print()
for mode in ['ode_hyper','iter_hyper']:
    s = t[t['mode']==mode].sort_values('pcc', ascending=False)
    print('=== %s (test PCC ?->?) ===' % mode)
    print(s[['config','pcc','mse','r2m','rlong']].round(4).to_string(index=False))
    print()
PY
