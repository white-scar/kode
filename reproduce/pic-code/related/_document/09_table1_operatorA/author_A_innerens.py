# -*- coding: utf-8 -*-
"""author_A_innerens.py - 用 inner-ens 的 R/phi 预测重构作者算子 A, 算 CV/val 精度"""
import numpy as np, pandas as pd, os, sys
DS=sys.argv[1] if len(sys.argv)>1 else "maize"
B="/data/fandata/dashuaige"
if DS=="maize":
    P=50; TRU=f"{B}/outputs/reproduce/author_tp1_maize/components"; IN=f"{B}/outputs/reproduce/author_tp1_maize_innerens/components"; OUT=f"{B}/outputs/reproduce/author_tp1_maize_innerens"
else:
    P=45; TRU=f"{B}/outputs/reproduce/author_tp1_arab/components"; IN=f"{B}/outputs/reproduce/author_tp1_arab_innerens/components"; OUT=f"{B}/outputs/reproduce/author_tp1_arab_innerens"
r=2
def buildA(Rf, Pf):
    Rm=Rf.reshape(r,r); Pm=Pf.reshape(P,r); return (Pm@Rm@np.linalg.pinv(Pm)).reshape(-1)
def acc(pred, truth):
    out=[]
    for j in range(pred.shape[1]):
        a=truth[:,j]; b=pred[:,j]
        if np.std(a)>1e-12 and np.std(b)>1e-12: out.append(np.corrcoef(a,b)[0,1])
    return float(np.nanmean(out))
rows=[]
for k in range(1,11):
    def rd(base, tag, comp):
        return pd.read_csv(f"{base}/iter{k:02d}_{tag}_{comp}.csv", index_col=0)
    # true train / val
    Rtr=rd(TRU,"trn","R"); Ptr=rd(TRU,"trn","phi"); Rva=rd(TRU,"val","R_true"); Pva=rd(TRU,"val","phi_true")
    # inner-ens predictions
    Rcv=rd(IN,"trn","R_cvpred"); Pcv=rd(IN,"trn","phi_cvpred")
    Rvl=rd(IN,"val","R_enspred"); Pvl=rd(IN,"val","phi_enspred")
    def mkA(Rd,Pd):
        out={}
        for g in Rd.index:
            out[g]=buildA(Rd.loc[g].values.astype(float), Pd.loc[g].values.astype(float))
        return pd.DataFrame(out).T
    A_tr=mkA(Rtr,Ptr); A_va=mkA(Rva,Pva)
    A_cv=mkA(Rcv,Pcv); A_vl=mkA(Rvl,Pvl)
    common_cv=[g for g in A_cv.index if g in A_tr.index]; common_vl=[g for g in A_vl.index if g in A_va.index]
    a_cv=acc(A_cv.loc[common_cv].values, A_tr.loc[common_cv].values)
    a_vl=acc(A_vl.loc[common_vl].values, A_va.loc[common_vl].values)
    rows.append(dict(iter=k, A_cv=a_cv, A_val=a_vl))
    print(f"iter{k}: A CV={a_cv:.3f}  A val={a_vl:.3f}", flush=True)
d=pd.DataFrame(rows); d.to_csv(f"{OUT}/author_A_innerens.csv", index=False)
print("\n== author A (tp1+MegaLMM, inner-ens) ==")
print(d[['A_cv','A_val']].agg(['mean','std']).round(4).to_string())
