# -*- coding: utf-8 -*-
"""kode_A_sig_trainval.py - 从 uv2 缓存算 A=U@V 与奇异值 sigma 的 train/val h2 + CV/ext 精度"""
import numpy as np, pandas as pd, os, sys
DS=sys.argv[1] if len(sys.argv)>1 else "maize"
B="/data/fandata/dashuaige"
if DS=="maize":
    GRMF=f"{B}/data/maize_magic/GRM_330x330_methodA_from552.csv"; CVF=f"{B}/data/maize_magic/cross_validation_folds_w_validation_set_maize.csv"; UV=f"{B}/outputs/reproduce/uv2"
else:
    GRMF=f"{B}/data/arabidopsis/arabidopsis_GRM_382.csv"; CVF=f"{B}/data/arabidopsis/cross_validation_folds_w_validation_set_arabidopsis.csv"; UV=f"{B}/outputs/reproduce/uv2_arab"
grm=pd.read_csv(GRMF,index_col=0); grm.index=grm.index.astype(str); grm.columns=grm.columns.astype(str)
cv=pd.read_csv(CVF,index_col=0); gids=list(grm.index); K=grm.loc[gids,gids].values.astype(float)
def h2b(Y,Kk,ng=120):
    n,E=Y.shape; w,U=np.linalg.eigh(Kk); w=np.clip(w,1e-8,None); Yt=U.T@Y; s1=U.sum(0)
    best=np.full(E,-np.inf); bh=np.zeros(E)
    for r in np.geomspace(1e-6,1e6,ng):
        d=w+r; yt2=np.sum(Yt**2/d[:,None],0); Bc=np.sum(Yt*(s1/d)[:,None],0)
        yPy=yt2-Bc**2/np.sum(s1**2/d); sg2=np.maximum(yPy/(n-1),1e-12)
        logL=-0.5*((n-1)*np.log(sg2)+np.sum(np.log(d))+np.log(np.sum(s1**2/d))+(n-1))
        m=logL>best; best[m]=logL[m]; bh[m]=1.0/(1.0+r)
    const=np.nanmax(np.abs(Y-Y.mean(0)),0)<1e-12; bh[const]=0
    return np.clip(bh,0,1)
def gcv(Y,Kk,seed=0,lam=1.0,nf=5):
    n,E=Y.shape; yc=Y-Y.mean(0); rng=np.random.RandomState(seed); idx=rng.permutation(n); fold=np.array_split(idx,nf)
    pred=np.zeros_like(yc)
    for te in fold:
        tr=np.setdiff1d(np.arange(n),te); pred[te]=Kk[np.ix_(te,tr)]@np.linalg.solve(Kk[np.ix_(tr,tr)]+lam*np.eye(len(tr)),yc[tr])
    out=[]
    for j in range(E):
        if np.std(yc[:,j])<1e-12 or np.std(pred[:,j])<1e-12: out.append(np.nan)
        else: out.append(np.corrcoef(yc[:,j],pred[:,j])[0,1])
    return np.nanmean(out)
def gext(Y,itr,iva,Kk,lam=1.0):
    ytr=Y[itr]-Y[itr].mean(0); pred=Kk[np.ix_(iva,itr)]@np.linalg.solve(Kk[np.ix_(itr,itr)]+lam*np.eye(len(itr)),ytr)
    out=[]
    for j in range(Y.shape[1]):
        a=Y[iva,j]; b=pred[:,j]
        out.append(np.corrcoef(a,b)[0,1] if np.std(a)>1e-12 and np.std(b)>1e-12 else np.nan)
    return np.nanmean(out)
rows=[]
for k in range(1,11):
    f=f"{UV}/gen_full_r2_f{k-1}.npz"
    if not os.path.exists(f): print("missing",f); continue
    z=np.load(f); U=z["U"]; V=z["V"]
    A=np.einsum("npr,nrq->npq",U,V).reshape(U.shape[0],-1)
    sig=np.stack([np.linalg.svd(U[i]@V[i],compute_uv=False) for i in range(U.shape[0])])
    folds=cv.iloc[:,k-1].values; itr=np.where(folds!=6)[0]; iva=np.where(folds==6)[0]
    rows.append(dict(iter=k,
      A_h2_tr=float(np.nanmean(h2b(A[itr],K[np.ix_(itr,itr)]))), A_h2_val=float(np.nanmean(h2b(A[iva],K[np.ix_(iva,iva)]))),
      A_cv=gcv(A,K), A_ext=gext(A,itr,iva,K),
      sig_h2_tr=float(np.nanmean(h2b(sig[itr],K[np.ix_(itr,itr)]))), sig_h2_val=float(np.nanmean(h2b(sig[iva],K[np.ix_(iva,iva)]))),
      sig_cv=gcv(sig,K), sig_ext=gext(sig,itr,iva,K)))
    print(f"iter{k}: A h2 {rows[-1]['A_h2_tr']:.3f}/{rows[-1]['A_h2_val']:.3f} cv {rows[-1]['A_cv']:.3f} ext {rows[-1]['A_ext']:.3f} | sig h2 {rows[-1]['sig_h2_tr']:.3f}/{rows[-1]['sig_h2_val']:.3f}", flush=True)
d=pd.DataFrame(rows); d.to_csv(f"{B}/outputs/reproduce/author_tp1_{DS}/kode_A_sig_trainval.csv",index=False)
print("\n== Kode %s A / sigma (10-fold mean) ==" % DS)
print(d.mean(numeric_only=True).round(4).to_string()); print("saved")
