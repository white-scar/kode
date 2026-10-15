#!/bin/bash
/root/miniconda3/bin/python - <<'PY'
import glob, os, pandas as pd
BASE='/root/autodl-tmp/qkn/outputs/kode_result'
DIRS={'???':'arab','??':'rice_indica_control','???RGB':'maize_rgb_drought'}
def grp(c):
    if c=='base': return 'base','base'
    if c.startswith('hid'): return 'hid',c[3:]
    if c.startswith('depth'): return 'depth',c[5:]
    if c.startswith('act_'): return 'act',c[4:]
    if c.startswith('seed'): return 'seed',c[4:]
    if c.startswith('dim'): return 'dim',c[3:]
    return '?',c
for name,d in DIRS.items():
    rows=[]
    for f in sorted(glob.glob(os.path.join(BASE,d,'*_all_summary_hp_*.csv'))):
        cfg=os.path.basename(f).split('all_summary_hp_')[-1].replace('.csv','')
        t=pd.read_csv(f)
        for _,r in t[t.split=='test'].iterrows():
            rows.append(dict(cfg=cfg,mode=r['mode'],pcc=float(r['pcc']),mse=float(r['MSE_norm']),r2=float(r['R2_median'])))
    df=pd.DataFrame(rows)
    df[['g','v']]=df.cfg.apply(lambda c: pd.Series(grp(c)))
    b_rec=df[(df.g=='base')&(df['mode']=='ode_hyper')].pcc.iloc[0]
    b_it=df[(df.g=='base')&(df['mode']=='iter_hyper')].pcc.iloc[0]
    print('='*70)
    print('=== %s?base: rec=%.4f iter=%.4f?==='%(name,b_rec,b_it))
    for g in ['hid','depth','act','seed']:
        sub=df[df.g==g]
        rec=sub[sub['mode']=='ode_hyper']; it=sub[sub['mode']=='iter_hyper']
        vals='/'.join(sorted(rec.v.unique(),key=lambda x:(len(x),x)))
        print('  %-6s %-18s rec %.4f~%.4f (?max %.4f) | iter %.4f~%.4f (?max %.4f)'%(
            g,vals,rec.pcc.min(),rec.pcc.max(),max(abs(rec.pcc-b_rec)),
            it.pcc.min(),it.pcc.max(),max(abs(it.pcc-b_it))))
        for v in sorted(rec.v.unique(),key=lambda x:(len(x),x)):
            rr=rec[rec.v==v].pcc.iloc[0]; ii=it[it.v==v].pcc.iloc[0]
            print('           %-6s rec=%.4f (%+.4f)  iter=%.4f (%+.4f)'%(v,rr,rr-b_rec,ii,ii-b_it))
    print()
PY
