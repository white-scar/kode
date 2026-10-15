#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""knode_missing_maize_arab.py - KNode missing-value experiment.
Train on kept timepoints only (masked loss), predict the held-out missing points.
Scenarios (maize): drop1/drop2/drop3/delblock; (arab): drop_last1/2/3, gap_mid.
"""
import math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from pathlib import Path
import warnings, time, argparse
warnings.filterwarnings('ignore')
torch.manual_seed(42); np.random.seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'outputs' / 'reproduce' / 'knode_missing_maize_arab'
OUT.mkdir(parents=True, exist_ok=True)

GRM_DIM = 50; RANK = 2; HID = 128; LR = 1e-3; WD = 1e-4

def load_data(dataset):
    if dataset == 'maize':
        pheno = pd.read_csv(BASE/'data'/'maize_magic'/'maize_trait_BLUP_50_NArmv_traits_afterClustering.csv')
        cv = pd.read_csv(BASE/'data'/'maize_magic'/'cross_validation_folds_w_validation_set_maize.csv', index_col=0)
        grm = pd.read_csv(BASE/'data'/'maize_magic'/'GRM_330x330_methodA_from552.csv', index_col=0)
    else:
        pheno = pd.read_csv(BASE/'data'/'arabidopsis'/'BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv')
        cv = pd.read_csv(BASE/'data'/'arabidopsis'/'cross_validation_folds_w_validation_set_arabidopsis.csv', index_col=0)
        grm = pd.read_csv(BASE/'data'/'arabidopsis'/'arabidopsis_GRM_382.csv', index_col=0)
    meta = ['bio_ID','accession','DAS']
    traits = [c for c in pheno.columns if c not in meta and not c.startswith('Unnamed') and c != '']
    traits = [t for t in traits if pheno[t].std() > 1e-12]
    P = len(traits)
    times = np.array(sorted(pheno['DAS'].unique()), dtype=np.float64)
    T = len(times)
    common = sorted(set(pheno['bio_ID'].astype(str)) & set(cv.index))
    grm.index = grm.index.astype(str); grm.columns = grm.columns.astype(str)
    grm = grm.loc[common, common].values.astype(np.float64)
    pca = PCA(n_components=GRM_DIM)
    grm_pca = pca.fit_transform(grm).astype(np.float32)
    Y = np.full((len(common), T, P), np.nan, dtype=np.float64)
    for pi, tr in enumerate(traits):
        sub = pheno[['bio_ID','DAS',tr]].copy(); sub['bio_ID'] = sub['bio_ID'].astype(str)
        Y[:,:,pi] = sub.pivot(index='bio_ID', columns='DAS', values=tr).loc[common].values
    assert not np.any(np.isnan(Y))
    tn = (times - times.min()) / (times.max() - times.min() + 1e-8)
    return dict(P=P, T=T, N=len(common), traits=traits, times=times, tn=tn, grm_pca=grm_pca, Y=Y, cv=cv)

def scenario_masks(dataset):
    if dataset == 'maize':
        # blocks of 5: DAS 15-19, 22-26, 29-33, 36-40, 43-47
        blocks = [[15,16,17,18,19],[22,23,24,25,26],[29,30,31,32,33],[36,37,38,39,40],[43,44,45,46,47]]
        cfg = {
            'drop1': dict(keep_per=4, extra=[47]),
            'drop2': dict(keep_per=3, extra=[46,47]),
            'drop3': dict(keep_per=2, extra=[45,46,47]),
            'delblock': dict(keep_blocks=4, extra=[43,44,45,46,47]),
        }
        scen = {}
        for name, c in cfg.items():
            if 'keep_blocks' in c:
                kept = [x for b in blocks[:c['keep_blocks']] for x in b]
            else:
                kept = [x for b in blocks for x in b[:c['keep_per']]]
            scen[name] = (kept, c['extra'])
        return scen
    else:
        # arab DAS 8..21
        scen = {
            'drop_last1': ([t for t in range(8,21)], [21]),
            'drop_last2': ([t for t in range(8,20)], [20,21]),
            'drop_last3': ([t for t in range(8,19)], [19,20,21]),
            'gap_mid': ([8,9,10,11,17,18,19,20,21], [12,13,14,15,16]),
        }
        return scen

class HyperODE(nn.Module):
    def __init__(self, P, cond_in, rank, hid):
        super().__init__()
        self.P=P; self.rank=rank
        self.enc=nn.Sequential(nn.Linear(cond_in,hid), nn.Tanh())
        self.hU=nn.Linear(hid,P*rank); self.hV=nn.Linear(hid,rank*P)
        self._U=self._V=None
    def set_z(self, grm_pca, y1):
        cond=torch.cat([grm_pca,y1],dim=-1)
        h=self.enc(cond); B=grm_pca.shape[0]
        self._U=self.hU(h).view(B,self.P,self.rank); self._V=self.hV(h).view(B,self.rank,self.P)
    def forward(self,t,h):
        return torch.bmm(torch.bmm(h.unsqueeze(1), self._V.transpose(1,2)), self._U.transpose(1,2)).squeeze(1)

def integrate(model, h0, ts):
    B,D=h0.shape; T=len(ts); traj=torch.zeros(B,T,D,device=h0.device,dtype=h0.dtype); traj[:,0]=h0
    h=h0
    for i in range(1,T):
        dt=max(ts[i].item()-ts[i-1].item(),1e-6)
        h=h+model(None,h)*dt; traj[:,i]=h
    return traj

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', default='maize', choices=['maize','arab'])
    ap.add_argument('--foldsets', default='1,7')
    ap.add_argument('--epochs', type=int, default=800)
    a=ap.parse_args()
    foldsets=[int(x) for x in a.foldsets.split(',') if x.strip()]
    D=load_data(a.dataset)
    P,T,N=D['P'],D['T'],D['N']; times=D['times']; tn=D['tn']; grm_pca=D['grm_pca']; Y=D['Y']; cv=D['cv']
    scen=scenario_masks(a.dataset)
    ts=torch.tensor(tn,dtype=torch.float32,device=DEVICE)

    rows=[]
    for foldset in foldsets:
        col=cv.iloc[:, foldset-1].values
        tr_pos=np.where((col>=1)&(col<=5))[0]; te_pos=np.where(col==6)[0]
        in_tr=tr_pos[(col[tr_pos]>=1)&(col[tr_pos]<=4)]; in_va=tr_pos[col[tr_pos]==5]
        tr_min=np.nanmin(Y[tr_pos],axis=(0,1)); tr_rng=np.maximum(np.nanmax(Y[tr_pos],axis=(0,1))-tr_min,1e-8)
        te_min=np.nanmin(Y[te_pos],axis=(0,1)); te_rng=np.maximum(np.nanmax(Y[te_pos],axis=(0,1))-te_min,1e-8)
        Ys_in=(Y[in_tr]-tr_min)/tr_rng; Ys_va=(Y[in_va]-tr_min)/tr_rng; Ys_te=(Y[te_pos]-te_min)/te_rng
        g_in=torch.tensor(grm_pca[in_tr],device=DEVICE); g_va=torch.tensor(grm_pca[in_va],device=DEVICE); g_te=torch.tensor(grm_pca[te_pos],device=DEVICE)

        for sname,(kept,extra) in scen.items():
            kept_idx=[i for i,t in enumerate(times) if t in kept]
            extra_idx=[i for i,t in enumerate(times) if t in extra]
            kept_t=torch.tensor([tn[i] for i in kept_idx],dtype=torch.float32,device=DEVICE)
            # keep the original full ts order for rollout (from t1)
            full_kept_sorted=sorted(kept_idx)
            model=HyperODE(P, GRM_DIM+P, RANK, HID).to(DEVICE)
            opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
            y_in=torch.tensor(Ys_in.astype(np.float32),device=DEVICE)
            y_va=torch.tensor(Ys_va.astype(np.float32),device=DEVICE)
            best=float('inf'); best_state=None
            for ep in range(a.epochs):
                model.train(); model.set_z(g_in, y_in[:,0,:])
                out=integrate(model, y_in[:,0,:], ts)          # full trajectory rollout from t1
                loss=F.mse_loss(out[:,kept_idx,:], y_in[:,kept_idx,:])
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                if ep%100==0 or ep==a.epochs-1:
                    model.eval()
                    with torch.no_grad():
                        model.set_z(g_va, y_va[:,0,:])
                        ov=integrate(model, y_va[:,0,:], ts)
                        vl=F.mse_loss(ov[:,kept_idx,:], y_va[:,kept_idx,:]).item()
                    if vl<best: best=vl; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                y_te=torch.tensor(Ys_te.astype(np.float32),device=DEVICE)
                # rec: roll from t1
                model.set_z(g_te, y_te[:,0,:]); pred_rec=integrate(model, y_te[:,0,:], ts)
                # iter: teacher-force on kept points, then roll extra from last kept
                pred_iter=torch.zeros_like(y_te); pred_iter[:,0,:]=y_te[:,0,:]
                for i in range(1,T):
                    dt=max(ts[i].item()-ts[i-1].item(),1e-6)
                    src = y_te[:,i-1,:] if (i-1) in kept_idx else pred_iter[:,i-1,:]
                    pred_iter[:,i,:]=src+model(None,src)*dt
            true_te=Y[te_pos]
            for mode,pred in [('rec',pred_rec.cpu().numpy()),('iter',pred_iter.cpu().numpy())]:
                # convert back to raw scale
                pred_raw=pred*te_rng+te_min
                for j in extra_idx:
                    vals=[]
                    for p in range(P):
                        r,_=pearsonr(true_te[:,j,p], pred_raw[:,j,p])
                        vals.append(r if not np.isnan(r) else np.nan)
                    rows.append(dict(dataset=a.dataset,foldset=foldset,scenario=sname,mode=mode,time=int(times[j]),pcc=float(np.nanmean(vals))))
            print(f'  fold{foldset} {sname} done', flush=True)

    df=pd.DataFrame(rows)
    tag=f"{a.dataset}_f{'_'.join(map(str,foldsets))}"
    df.to_csv(OUT/f'knode_missing_{tag}.csv', index=False)
    print(df.groupby(['scenario','mode'])['pcc'].mean().round(4).to_string())
    print('saved ->', OUT/f'knode_missing_{tag}.csv')

if __name__=='__main__':
    main()
