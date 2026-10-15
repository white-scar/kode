# -*- coding: utf-8 -*-
"""baselines_per_timepoint.py - 从 _preds.npz 重算逐时间点 PCC/MSE"""
import numpy as np, pandas as pd, sys
from pathlib import Path
BASE=Path("/data/fandata/dashuaige")
DS=sys.argv[1] if len(sys.argv)>1 else "maize"
if DS=="maize":
    pheno=pd.read_csv(BASE/"data/maize_magic/maize_trait_BLUP_50_NArmv_traits_afterClustering.csv")
    cv=pd.read_csv(BASE/"data/maize_magic/cross_validation_folds_w_validation_set_maize.csv",index_col=0)
    npz=BASE/"outputs/reproduce/knode_dynamic_baselines_full_preds.npz"
    out=BASE/"outputs/reproduce/knode_dynamic_baselines_full_per_timepoint.csv"
else:
    pheno=pd.read_csv(BASE/"data/arabidopsis/BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv")
    cv=pd.read_csv(BASE/"data/arabidopsis/cross_validation_folds_w_validation_set_arabidopsis.csv",index_col=0)
    npz=BASE/"outputs/reproduce/knode_dynamic_baselines_full_arab_preds.npz"
    out=BASE/"outputs/reproduce/knode_dynamic_baselines_full_arab_per_timepoint.csv"
meta=["bio_ID","accession","DAS"]
traits=[c for c in pheno.columns if c not in meta and not c.startswith("Unnamed") and c!=""]
traits=[t for t in traits if pheno[t].std()>1e-12]
times=sorted(pheno["DAS"].unique()); T=len(times); P=len(traits)
common=sorted(set(pheno["bio_ID"].astype(str))&set(cv.index))
gids=common
Y=np.full((len(gids),T,P),np.nan)
for pi,tr in enumerate(traits):
    sub=pheno[["bio_ID","DAS",tr]].copy(); sub["bio_ID"]=sub["bio_ID"].astype(str)
    Y[:,:,pi]=sub.pivot(index="bio_ID",columns="DAS",values=tr).loc[gids].values
z=np.load(npz)
fams=sorted(set(k.split("_iter")[0] for k in z.keys()))
print("fams:",fams)
rows=[]
for k in range(10):
    fc=cv.loc[gids].iloc[:,k].values.astype(int)
    tr=np.where((fc>=1)&(fc<=5))[0]; te=np.where(fc==6)[0]
    tr_min=np.nanmin(Y[tr],axis=(0,1)); tr_rng=np.maximum(np.nanmax(Y[tr],axis=(0,1))-tr_min,1e-8)
    Yte=((Y[te]-tr_min)/tr_rng)
    for fam in fams:
        key=f"{fam}_iter{k}_avg"
        if key not in z: continue
        pred=z[key]
        for t in range(1,T):
            rs=[np.corrcoef(pred[:,t,p],Yte[:,t,p])[0,1] for p in range(P)
                if np.std(pred[:,t,p])>1e-12 and np.std(Yte[:,t,p])>1e-12]
            mse=float(np.mean((pred[:,t,:]-Yte[:,t,:])**2))
            rows.append(dict(fam=fam,iter=k,time=int(times[t]),pcc=float(np.nanmean(rs)),mse=mse))
d=pd.DataFrame(rows); d.to_csv(out,index=False)
g=d.groupby(["fam","time"]).agg(pcc=("pcc","mean"),mse=("mse","mean")).reset_index()
print(g.pivot(index="time",columns="fam",values="pcc").round(3).to_string())
print("\nsaved ->",out)
