#!/bin/bash
/root/miniconda3/bin/python - <<'PY'
import glob, os, pandas as pd
BASE='/root/autodl-tmp/qkn/outputs/kode_result'
DIRS={'???':'arab','??':'rice_indica_control','???RGB':'maize_rgb_drought'}
for name, d in DIRS.items():
    rows=[]
    for f in sorted(glob.glob(os.path.join(BASE,d,'*_all_summary_hp_*.csv'))):
        cfg=os.path.basename(f).split('all_summary_hp_')[-1].replace('.csv','')
        t=pd.read_csv(f)
        for _,r in t[t.split=='test'].iterrows():
            rows.append(dict(cfg=cfg,mode=r['mode'],pcc=float(r['pcc']),mse=float(r['MSE_norm']),r2=float(r['R2_median'])))
    df=pd.DataFrame(rows)
    if df.empty:
        print('=== %s: ???' % name); continue
    print('='*62)
    print('=== %s  ???=%d  mode=%s' % (name, df.cfg.nunique(), sorted(df['mode'].unique())))
    for mode,lab in [('ode_hyper','recursive'),('iter_hyper','iterative')]:
        s=df[df['mode']==mode].sort_values('pcc',ascending=False)
        if s.empty: continue
        print('  --- %s ---' % lab)
        for _,r in s.iterrows():
            mark='  <-- base' if r['cfg']=='base' else ''
            print('     %-11s PCC=%.4f  MSE=%.5f  R2med=%+.4f%s' % (r['cfg'],r['pcc'],r['mse'],r['r2'],mark))
    print()
PY
