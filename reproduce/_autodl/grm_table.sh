#!/bin/bash
/root/miniconda3/bin/python - <<'PY'
import glob, os, pandas as pd
BASE='/root/autodl-tmp/qkn/outputs/kode_result'
DIRS={'???':'arab','??':'rice_indica_control','???RGB':'maize_rgb_drought'}
for name,d in DIRS.items():
    rows=[]
    for f in sorted(glob.glob(os.path.join(BASE,d,'*_all_summary_hp_*.csv'))):
        cfg=os.path.basename(f).split('all_summary_hp_')[-1].replace('.csv','')
        if not (cfg.startswith('dim') or cfg=='base'): continue
        t=pd.read_csv(f)
        for _,r in t[t.split=='test'].iterrows():
            rows.append(dict(cfg=cfg,mode=r['mode'],pcc=float(r['pcc']),mse=float(r['MSE_norm']),r2=float(r['R2_median'])))
    df=pd.DataFrame(rows)
    df['dim']=df.cfg.apply(lambda c: 999 if c=='base' else int(c.replace('dim','')))
    df=df.sort_values(['dim','mode'])
    print('='*66)
    print('=== %s ? GRM ??(D_G) ???10 fold-set, test?'%name)
    for _,r in df.iterrows():
        lab='recursive' if r['mode']=='ode_hyper' else 'iterative'
        dg='base(50/30/20)' if r['cfg']=='base' else 'D_G=%d'%int(r['dim'])
        mark='   <-- base(??)' if r['cfg']=='base' else ''
        print('  %-14s %-10s PCC=%.4f  MSE=%.5f  R2med=%+.4f%s'%(dg,lab,r['pcc'],r['mse'],r['r2'],mark))
    sub=df[(df['mode']=='ode_hyper')&(df['cfg']!='base')]
    if len(sub):
        b=sub.loc[sub['pcc'].idxmax()]
        plat=sub[sub['pcc']>=b['pcc']-0.005]
        print('  -> recursive ?? D_G=%d (%.4f)???(?0.005) = %s'%(b['dim'],b['pcc'],sorted(int(x) for x in plat['dim'])))
    else:
        print('  -> ??? base ???')
    print()
PY
