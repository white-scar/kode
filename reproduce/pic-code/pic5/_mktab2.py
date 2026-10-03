import pandas as pd, numpy as np
b=pd.read_csv("fig_baselines_data.csv")
old=pd.read_csv("table_dynamic_methods_overall.csv")
DS={"maize":"maize MAGIC","arab":"Arabidopsis","rice":"rice (control)","uav":"UAV RGB (drought)"}
M ={"Kode_iter":"Kode (iterative)","Kode_rec":"Kode (recursive)","ststp":"ST-STP","lstm":"LSTM",
    "ar_transformer":"ARTransformer","ode_rnn":"ODE-RNN","GRU-ODE-Style":"GRU-ODE-Style"}
rows=[]
for ds in ["maize","arab","rice","uav"]:
    d=b[b.dataset==ds]
    tsp=sorted(d.time.unique()); drng="%d-%d"%(tsp[0],tsp[-1])
    sub=[]
    for m,drop in d.groupby("method"):
        sub.append(dict(dataset=DS[ds], method=M[m], PCC=drop.pcc.mean(), PCC_sd=drop.pcc_sd.mean(),
                        MSE=drop.mse.mean(), MSE_sd=drop.mse_sd.mean(), n_time=len(drop), DAS=drng))
    sub=sorted(sub,key=lambda r:-r["PCC"])
    for k,r in enumerate(sub,1): r["rank_PCC"]=k
    rows+=sub
new=pd.DataFrame(rows)[["dataset","method","PCC","PCC_sd","MSE","MSE_sd","n_time","DAS","rank_PCC"]]
new.to_csv("table_dynamic_methods_overall.csv",index=False)
# diff vs old
chg=0
for _,r in new.iterrows():
    o=old[(old.dataset==r.dataset)&(old.method==r.method)]
    if len(o)==0: print("NEW ROW", r.dataset, r.method); chg+=1; continue
    o=o.iloc[0]
    if abs(o.PCC-r.PCC)>1e-9 or abs(o.MSE-r.MSE)>1e-9:
        print("CHANGED %-18s %-18s PCC %.6f -> %.6f | MSE %.6f -> %.6f"%(r.dataset,r.method,o.PCC,r.PCC,o.MSE,r.MSE)); chg+=1
print("changed rows:",chg,"| total rows:",len(new))
