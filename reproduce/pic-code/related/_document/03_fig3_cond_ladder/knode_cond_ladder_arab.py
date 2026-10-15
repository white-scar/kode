# -*- coding: utf-8 -*-
"""knode_cond_ladder_arab.py — Fig.3 cond ladder (条件信息逐层增加): 拟南芥 (arabidopsis): data/arabidopsis/

协议与玉米 Kode 主方法逐位一致 (本脚本主体逐行来自该模板, 未改模型/训练/指标/聚合逻辑):
    10 fold-set; fold1-5 训练 (5 折 inner 集成: 每模型 4/5 训练 + 1/5 validation); fold6 测试取 5 个模型平均。
    checkpoint selection: 跑满 1000 epochs, 每 50 epochs 在验证集上算一次 MSE, 保留验证误差最低的那份权重
                          (没有 patience-based early stopping, 不存在提前终止)。
    模型: c_i = [z_i(D_G), y_i(t1)] -> Linear+tanh -> U_i, V_i, b_i;  A_i = U_i V_i;
          dh_i/dt = h_i A_i^T + b_i  (affine drift b_i 保留);
          归一化真实时间间隔上的 explicit Euler; 训练 loss = 整轨迹 MSE; 评估指标剔除 t1。
    HID=128, Tanh, Adam lr=1e-3, weight decay=1e-4, grad clip=1.0, 1000 epochs。
    唯一变化: 生成算子 A_i 的条件信息 (persistence / shared / grm / tp1 / full), 其余全部相同。

锁定配置 (2026-09-19, 由 D_G x rank 联合扫描的人工判断确定):
    dataset = arab, D_G = 30, rank = 2
    --grm_dim / --rank 可覆盖 (后续做更高 rank 的 cond ladder 时用), 默认即上面的锁定值。

数据身份硬检查: N=382, T=14, P=45; 通过后写 dataset_manifest.json。

冒烟隔离 (防"冒烟覆盖正式结果"):
    --smoke 强制 1 fold x 50 epochs, 结果全部写进 run_dir/smoke/ 子目录;
    正式结果只允许 n_epochs=1000, 且 aggregate() 只读自己 run_dir 下的 f[0-9].csv。

输出 (每个 (数据集, 环境, D_G, rank) 一个独立 run directory, 不同配置永不互相覆盖):
    outputs/reproduce/cond_ladder_<dataset>[_<env>]_dg<D_G>_r<rank>/
        f{fold}.csv        逐折条件级 (cond, pcc, mse, nrmse, r2, sec + dataset/grm_dim/rank/n_epochs)
        trait_f{fold}.csv  逐折 x 时点 x 性状原始长表
        agg.csv            条件级汇总 (pcc_mean/std, mse_mean/std, nrmse, r2, n)
        dataset_manifest.json / run_meta.json
        smoke/             --smoke 的结果 (f0.csv / trait_f0.csv / agg.csv / run_meta.json)
"""
import math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr
from pathlib import Path
import argparse, warnings, time, json
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # reproduce/ 在上一级: knode_perfect_vs_predicted_operators 在那里
import knode_rank_genetics_value_arab as M

torch.manual_seed(42); np.random.seed(42)   # 与主方法一致: 只在模块级设一次

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 1e-3; WEIGHT_DECAY = 1e-4; HID = 128
GRM_DIM_DEFAULT, RANK_DEFAULT = 30, 2     # 本数据集锁定值 (2026-09-19); --grm_dim/--rank 可覆盖
GRM_DIM, RANK = GRM_DIM_DEFAULT, RANK_DEFAULT
N_EPOCHS_DEFAULT = 1000                    # 正式结果只允许这个值 (短跑走 --smoke)
DATASET = 'arab'
EXP_CFG = {'drought': (382, 14, 45)}   # 数据身份硬检查 (与 dg_rank joint 脚本一致)
EXP_N, EXP_T, EXP_P = EXP_CFG['drought']
ENV = 'drought'                            # 只有 RGB 用 (drought/irrigated); arab/rice 固定 drought
SMOKE = False                              # True -> 所有输出走 *_smoke_*, 绝不覆盖正式结果
CONDS = ['persistence', 'shared', 'grm', 'tp1', 'full']
def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").is_dir() and (p / "outputs").is_dir():
            return p
    raise SystemExit("找不到 knode 根目录 (需要同时存在 data/ 与 outputs/)")
ROOT = _find_root(Path(__file__).resolve().parent)
OUT_DIR = ROOT / 'outputs' / 'reproduce'; OUT_DIR.mkdir(parents=True, exist_ok=True)
P = T = N = None
_MANIFEST_WRITTEN = False


def set_dims(grm_dim, rank):
    """把命令行给的 D_G / rank 落到模块全局 (CondLadderModel 与 channels 都读这两个全局)。"""
    global GRM_DIM, RANK
    GRM_DIM, RANK = int(grm_dim), int(rank)


def run_dir():
    """每个 (数据集, 环境, D_G, rank) 一个独立 run directory: 不同配置的结果永不互相覆盖/混杂。
    --smoke 再下沉到 smoke/ 子目录, 所以 aggregate 永远不会读到冒烟结果。"""
    name = f'cond_ladder_{DATASET}' + ('' if ENV == 'drought' else '_' + ENV) + f'_dg{GRM_DIM}_r{RANK}'
    d = OUT_DIR / name
    if SMOKE:
        d = d / 'smoke'
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_path(kind, fold=None):
    d = run_dir()
    if kind == 'trait':
        return d / f'trait_f{fold}.csv'
    if kind == 'fold':
        return d / f'f{fold}.csv'
    return d / 'agg.csv'


def write_manifest():
    global _MANIFEST_WRITTEN
    if _MANIFEST_WRITTEN:
        return
    manifest = dict(dataset=DATASET, environment=ENV, N=int(N), T=int(T), P=int(P),
                    DAS=[float(x) for x in M.times_all], trait_names=list(M.trait_cols),
                    cv_foldsets=int(M.cv_foldsets.shape[1]))
    p = run_dir() / 'dataset_manifest.json'
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    _MANIFEST_WRITTEN = True
    print(f'[data] saved -> {p}', flush=True)


def write_run_meta(folds, epochs):
    meta = dict(dataset=DATASET, environment=ENV, condition_list=CONDS,
                D_G=int(GRM_DIM), rank=int(RANK), epochs=int(epochs), folds=folds,
                smoke=bool(SMOKE), hidden=HID, activation='tanh', drift='affine',
                optimizer='adam', lr=LR, weight_decay=WEIGHT_DECAY, grad_clip=1.0,
                solver='explicit_euler_normalized_real_gaps', loss='mse_full_trajectory',
                checkpoint_metric='validation_mse_excluding_t1', inner_folds=5, outer_foldsets=10,
                checkpoint_selection='best validation MSE over every-50-epoch checks; full 1000 epochs, no patience-based early stopping',
                early_stopping=False,
                seed=42, ensemble='5 inner models averaged on fold6', eval_excludes_t1=True,
                rng='reset torch/np seed at the start of every condition',
                time=time.strftime('%Y-%m-%d %H:%M:%S'))
    p = run_dir() / 'run_meta.json'
    p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[meta] saved -> {p}', flush=True)


def check_data():
    """硬检查数据身份 (N/T/P), 通过后写 dataset_manifest.json。"""
    eN, eT, eP = EXP_CFG[ENV]
    assert P == eP, f'Unexpected trait number: P={P}, expected {eP}'
    assert T == eT, f'Unexpected time points: T={T}, expected {eT}'
    assert N == eN, f'Unexpected genotype number: N={N}, expected {eN}'
    print(f'[data] 硬检查通过: N={N} T={T} P={P} | D_G={GRM_DIM} rank={RANK} env={ENV}', flush=True)
    write_manifest()


class CondLadderModel(nn.Module):
    def __init__(self, P, rank=RANK, hid=HID):
        super().__init__()
        self.P = P; self.rank = rank; self.cond_in = GRM_DIM + P
        self.hyper_enc = nn.Sequential(nn.Linear(self.cond_in, hid), nn.Tanh())
        self.hyper_U = nn.Linear(hid, P*rank); self.hyper_V = nn.Linear(hid, rank*P)
        self.hyper_b = nn.Linear(hid, P)
        self._U_dyn = self._V_dyn = self._b_dyn = None
    def set_z(self, g, t):
        if g is not None and not torch.is_tensor(g): g = torch.tensor(g, device=DEVICE)
        if t is not None and not torch.is_tensor(t): t = torch.tensor(t, device=DEVICE)
        B = g.shape[0] if g is not None else t.shape[0]
        g = g if g is not None else torch.zeros(B, GRM_DIM, device=DEVICE)
        t = t if t is not None else torch.zeros(B, self.P, device=DEVICE)
        h = self.hyper_enc(torch.cat([g,t], dim=-1))
        self._U_dyn = self.hyper_U(h).view(B,self.P,self.rank)
        self._V_dyn = self.hyper_V(h).view(B,self.rank,self.P)
        self._b_dyn = self.hyper_b(h).view(B,self.P)
    def forward(self, t, h):
        lin = torch.bmm(torch.bmm(h.unsqueeze(1), self._V_dyn.transpose(1,2)), self._U_dyn.transpose(1,2)).squeeze(1)
        return lin + self._b_dyn

def channels(cond, ids):
    ids=np.asarray(ids)
    z=torch.tensor(M.grm_pca[ids],device=DEVICE); y=torch.tensor(M.y1_all[ids],device=DEVICE)
    if cond=='full': return z,y
    if cond=='grm':  return z,None
    if cond=='tp1':  return None,y
    if cond=='shared': return torch.zeros(len(ids),GRM_DIM,device=DEVICE), torch.zeros(len(ids),P,device=DEVICE)
    raise ValueError(cond)

def euler(model,h0,ts):
    B,D=h0.shape; TL=len(ts); tr=torch.zeros(B,TL,D,device=h0.device,dtype=h0.dtype); tr[:,0]=h0; h=h0
    for i in range(1,TL):
        dt=max(ts[i].item()-ts[i-1].item(),1e-6); h=h+model(None,h)*dt; tr[:,i]=h
    return tr

def train_model(cond, Ytr, tr_ids, Yva, va_ids, seed, epochs):
    m=CondLadderModel(P,RANK,HID).to(DEVICE); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    y_tr=torch.tensor(Ytr.astype(np.float32),device=DEVICE); g_tr,t_tr=channels(cond,tr_ids)
    y_va=torch.tensor(Yva.astype(np.float32),device=DEVICE); g_va,t_va=channels(cond,va_ids)
    best=None; bl=float('inf')
    for ep in range(epochs):
        m.train(); m.set_z(g_tr,t_tr)
        pred=euler(m,y_tr[:,0,:],M.times_t)
        loss=F.mse_loss(pred,y_tr); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if ep%50==0 or ep==epochs-1:
            with torch.no_grad():
                m.eval(); m.set_z(g_va,t_va)
                pv=euler(m,y_va[:,0,:],M.times_t)
                vl=F.mse_loss(pv[:,1:,:],y_va[:,1:,:]).item()
            if not math.isnan(vl) and vl<bl: bl=vl; best={k:v.cpu().clone() for k,v in m.state_dict().items()}
    if best is None: best={k:v.cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best); m.eval(); return m

def rollout_pred(m, cond, ids):
    g,t=channels(cond,ids)
    with torch.no_grad():
        m.set_z(g,t); h0=torch.tensor(M.y1_all[ids],device=DEVICE)
        pred=euler(m,h0,M.times_t)
    return pred.cpu().numpy()

def metrics(pred, Ytrue):
    pccs=[]; trows=[]
    for t in range(1,T):
        rs=[]
        for p in range(P):
            y=Ytrue[:,t,p]; pr=pred[:,t,p]
            r=float(pearsonr(pr,y)[0]) if np.std(y)>1e-12 else np.nan
            rs.append(r)
            trows.append(dict(timepoint=int(M.times_all[t]), trait=M.trait_cols[p],
                              pcc=r, mse=float(np.mean((pr-y)**2))))
        pccs.append(float(np.mean(rs)) if rs else np.nan)
    mse=float(np.mean((pred[:,1:,:]-Ytrue[:,1:,:])**2))
    return float(np.nanmean(pccs)), mse, float(np.sqrt(mse)), float(1-mse/np.var(Ytrue[:,1:,:])), trows

def run_fold(fold, epochs):
    global P,T,N
    M.load_data(GRM_DIM, fold); P,T,N=M.P,M.T,M.N
    check_data()
    fc=M.cv_foldsets[:,fold]
    tr_idx=np.where((fc>=1)&(fc<=5))[0]; te_idx=np.where(fc==6)[0]
    Ys_te=((M.Y_all[te_idx]-M.tr_min)/M.tr_rng).astype(np.float32)
    rows=[]; trait_rows=[]
    for cond in CONDS:
        t0=time.time()
        if cond=='persistence':
            y1=M.y1_all[te_idx]; pred=np.repeat(y1[:,None,:],T,axis=1)
            pcc,mse,nrmse,r2,trows=metrics(pred,Ys_te)
            for _tr in trows: _tr.update(cond=cond, fold=fold, dataset=DATASET,
                                      grm_dim=GRM_DIM, rank=RANK, n_epochs=epochs)
            trait_rows.extend(trows)
        else:
            torch.manual_seed(42); np.random.seed(42)   # 每个条件从同一 RNG 状态开始
            acc=[]  # 累积各 inner 模型对 test 的预测
            for inner in range(1,6):
                itr=tr_idx[fc[tr_idx]!=inner]; iva=tr_idx[fc[tr_idx]==inner]
                Ytr=((M.Y_all[itr]-M.tr_min)/M.tr_rng).astype(np.float32)
                Yva=((M.Y_all[iva]-M.tr_min)/M.tr_rng).astype(np.float32)
                m=train_model(cond,Ytr,itr,Yva,iva,42,epochs)
                acc.append(rollout_pred(m,cond,te_idx))
            pred=np.mean(acc,axis=0)
            pcc,mse,nrmse,r2,trows=metrics(pred,Ys_te)
            for _tr in trows: _tr.update(cond=cond, fold=fold, dataset=DATASET,
                                      grm_dim=GRM_DIM, rank=RANK, n_epochs=epochs)
            trait_rows.extend(trows)
        rows.append(dict(cond=cond,fold=fold,pcc=round(pcc,4),mse=round(mse,6),nrmse=round(nrmse,4),r2=round(r2,4),sec=round(time.time()-t0,1),
                         dataset=DATASET,grm_dim=GRM_DIM,rank=RANK,n_epochs=epochs))
        print(f'  [{cond}] pcc={pcc:.4f} mse={mse:.5f} ({time.time()-t0:.0f}s)',flush=True)
    pd.DataFrame(trait_rows).to_csv(out_path('trait', fold),index=False)
    d=pd.DataFrame(rows); d.to_csv(out_path('fold', fold),index=False); return d

def aggregate():
    fs=sorted(run_dir().glob('f[0-9].csv'))
    if not fs: print('none'); return
    d=pd.concat([pd.read_csv(f) for f in fs])
    g=d.groupby('cond').agg(pcc_mean=('pcc','mean'),pcc_std=('pcc','std'),mse_mean=('mse','mean'),
                            nrmse_mean=('nrmse','mean'),r2_mean=('r2','mean'),n=('pcc','count')).reset_index()
    o={c:i for i,c in enumerate(CONDS)}; g['_o']=g.cond.map(o); g=g.sort_values('_o').drop(columns='_o')
    g['dataset']=DATASET; g['grm_dim']=GRM_DIM; g['rank']=RANK
    g.to_csv(out_path('agg'),index=False); print(g.round(4).to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--folds',type=str,default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--n_epochs',type=int,default=N_EPOCHS_DEFAULT)
    ap.add_argument('--aggregate',action='store_true')
    ap.add_argument('--smoke',action='store_true',help='冒烟: 1 fold x 50 epochs, 只写 run_dir/smoke/, 不碰正式结果')
    ap.add_argument('--grm_dim',type=int,default=GRM_DIM_DEFAULT,help='GRM PCA 维度 (默认=本数据集锁定值)')
    ap.add_argument('--rank',type=int,default=RANK_DEFAULT,help='算子 rank (默认=本数据集锁定值)')
    ap.add_argument('--env',type=str,default='drought',choices=['drought','irrigated'],help='仅 RGB 用')
    a=ap.parse_args()
    set_dims(a.grm_dim, a.rank)
    if a.env != 'drought':
        if not hasattr(M, 'ENV'):
            raise SystemExit(f'{DATASET} 只有一个环境, 不支持 --env {a.env}')
        ENV = a.env; M.ENV = a.env          # 数据模块按模块级 ENV 选表型/折文件
    if a.smoke:
        SMOKE = True
        folds = [0]        # 必须是 int: fold_idx() 拿它做列索引
        epochs = 50 if a.n_epochs == N_EPOCHS_DEFAULT else a.n_epochs
    else:
        if a.n_epochs != N_EPOCHS_DEFAULT:
            raise SystemExit(f'正式结果只允许 {N_EPOCHS_DEFAULT} epochs (防短跑覆盖正式结果); 短跑请加 --smoke')
        folds = [int(x) for x in a.folds.split(',') if x.strip()]
        epochs = a.n_epochs
    print(f'Device={DEVICE} | dataset={DATASET} env={ENV} | D_G={GRM_DIM} rank={RANK} | '
          f'epochs={epochs} | folds={folds} | smoke={SMOKE}', flush=True)
    if a.aggregate:
        aggregate()
    else:
        write_run_meta(folds, epochs)
        for fd in folds:
            print(f'===== fold {fd} =====',flush=True); run_fold(fd,epochs)
        aggregate()
