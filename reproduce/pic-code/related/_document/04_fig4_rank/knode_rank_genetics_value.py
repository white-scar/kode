#!/usr/bin/env python3
"""knode_rank_genetics_value.py ? "??? != ??? != ??" formal table (fixed 100-dim masked input).

For each (cond, rank) train the hyper-ODE (predicted stage: inner 5-fold (4/5 train + 1/5 early-stop) -> ensemble,
extract operator for all N, rollout fold6), then compute three quantities:
  1. heritability : GREML h2 of U/V entries (mean over entries)
  2. predictability: GBLUP 5-fold CV (within) + GBLUP train->test extrapolation of U/V entries
  3. value        : end-to-end rollout PCC / MSE on the held-out fold

conditions (all cond_in=100, channel masking):
  full=[grm,y1]  grmonly=[grm,0]  tp1only=[0,y1]  const=[0,0]  noise=[noise,noise]

Usage (from reproduce/):
  python knode_rank_genetics_value.py --fold 0 --n_epochs 1000
Outputs:
  outputs/reproduce/knode_rank_genetics_value_f{fold}.csv
  outputs/reproduce/<cache>/gen_{cond}_r{rank}_f{fold}_i{inner}.npz   (checkpoint; 已算过的 cell 直接复用, 不重训)
"""
import math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr
from pathlib import Path
import argparse, warnings, time
warnings.filterwarnings('ignore')
import knode_perfect_vs_predicted_operators as M

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 1e-3; WD = 1e-4; HID = 128; GRM_DIM = 50
CONDS = ['full', 'grmonly', 'tp1only', 'const', 'noise']
RANKS = [1, 2, 3, 4, 6, 8, 10, 16, 20, 30, 50]
OUT = Path('../outputs/reproduce'); OUT.mkdir(parents=True, exist_ok=True)
UV2 = OUT / 'uv3'; UV2.mkdir(exist_ok=True)

torch.manual_seed(42); np.random.seed(42)

P = T = N = None
noise2_all = None


class MaskedHyperODEModel(nn.Module):
    def __init__(self, P, rank, hid=128):
        super().__init__()
        self.P = P; self.rank = rank
        self.cond_in = GRM_DIM + P
        self.hyper_enc = nn.Sequential(nn.Linear(self.cond_in, hid), nn.Tanh())
        self.hyper_U = nn.Linear(hid, P * rank)
        self.hyper_V = nn.Linear(hid, rank * P)
        self.hyper_b = nn.Linear(hid, P)          # genotype-specific affine drift b_i
        self._U_dyn = self._V_dyn = self._b_dyn = None

    def set_z(self, grm_chan, t1_chan):
        if grm_chan is not None and not torch.is_tensor(grm_chan):
            grm_chan = torch.tensor(grm_chan, device=DEVICE)
        if t1_chan is not None and not torch.is_tensor(t1_chan):
            t1_chan = torch.tensor(t1_chan, device=DEVICE)
        B = grm_chan.shape[0] if grm_chan is not None else t1_chan.shape[0]
        g = grm_chan if grm_chan is not None else torch.zeros(B, GRM_DIM, device=DEVICE)
        t = t1_chan if t1_chan is not None else torch.zeros(B, self.P, device=DEVICE)
        cond = torch.cat([g, t], dim=-1)
        h = self.hyper_enc(cond)
        self._U_dyn = self.hyper_U(h).view(B, self.P, self.rank)
        self._V_dyn = self.hyper_V(h).view(B, self.rank, self.P)
        self._b_dyn = self.hyper_b(h).view(B, self.P)

    def forward(self, t, h):
        lin = torch.bmm(torch.bmm(h.unsqueeze(1), self._V_dyn.transpose(1, 2)),
                        self._U_dyn.transpose(1, 2)).squeeze(1)
        return lin + self._b_dyn


def euler_integrate(model, h0, ts):
    B, D = h0.shape; T_len = len(ts)
    traj = torch.zeros(B, T_len, D, device=h0.device, dtype=h0.dtype); traj[:, 0] = h0
    h = h0
    for i in range(1, T_len):
        dt = max(ts[i].item() - ts[i - 1].item(), 1e-6)
        h = h + model(None, h) * dt
        traj[:, i] = h
    return traj


def channels(cond, ids):
    ids = np.asarray(ids)
    if cond == 'full':
        g = torch.tensor(M.grm_pca[ids], device=DEVICE); t = torch.tensor(M.y1_all[ids], device=DEVICE)
    elif cond == 'grmonly':
        g = torch.tensor(M.grm_pca[ids], device=DEVICE); t = None
    elif cond == 'tp1only':
        g = None; t = torch.tensor(M.y1_all[ids], device=DEVICE)
    elif cond == 'const':
        g = torch.zeros(len(ids), GRM_DIM, device=DEVICE); t = torch.zeros(len(ids), P, device=DEVICE)   # 无个体信息输入: 全 0 (与图三 shared 同定义)
    elif cond == 'noise':
        g = torch.tensor(M.noise_all[ids], device=DEVICE); t = torch.tensor(noise2_all[ids], device=DEVICE)
    else:
        raise ValueError(cond)
    return g, t


def train_model(cond, rank, Ys_tr, train_ids, Ys_early, early_ids, n_epochs):
    m = MaskedHyperODEModel(P, rank, HID).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    y_tr = torch.tensor(Ys_tr.astype(np.float32), device=DEVICE)
    g_tr, t_tr = channels(cond, train_ids)
    y_early = torch.tensor(Ys_early.astype(np.float32), device=DEVICE)
    g_early, t_early = channels(cond, early_ids)
    best = None; best_loss = float('inf')
    for ep in range(n_epochs):
        m.train(); m.set_z(g_tr, t_tr)
        pred = euler_integrate(m, y_tr[:, 0, :], M.times_t)
        loss = F.mse_loss(pred, y_tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 50 == 0 or ep == n_epochs - 1:
            with torch.no_grad():
                m.eval(); m.set_z(g_early, t_early)
                pv = euler_integrate(m, y_early[:, 0, :], M.times_t)
                vl = F.mse_loss(pv[:, 1:, :], y_early[:, 1:, :]).item()
            if not math.isnan(vl) and vl < best_loss:
                best_loss = vl; best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    if best is None:
        best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best); m.eval()
    return m


def extract_uv(m, cond):
    ids = np.arange(N)
    g, t = channels(cond, ids)
    with torch.no_grad():
        m.set_z(g, t)
        U = m._U_dyn.cpu().numpy().copy()
        V = m._V_dyn.cpu().numpy().copy()
    return U, V


def cum_energy(K):
    """Cumulative spectral energy share of the operator K = U V, averaged over genotypes."""
    sv = np.linalg.svd(K, compute_uv=False)
    esq = sv ** 2
    return np.cumsum(esq, axis=1) / esq.sum(1)[:, None]


def predict_torch(m, cond, ids):
    """Predict exactly like the main script: set_z(cond) -> explicit Euler rollout from the normalized t1."""
    g, t = channels(cond, ids)
    with torch.no_grad():
        m.eval(); m.set_z(g, t)
        y1 = torch.tensor(M.y1_all[ids], device=DEVICE)
        out = euler_integrate(m, y1, M.times_t)
    return out.cpu().numpy()


def roll_pred(K, ids):
    """Numpy rollout of the extracted operator K = U V (explicit Euler, same dt as the main script)."""
    y1 = M.y1_all[ids]
    pred = np.zeros((len(ids), T, P), dtype=np.float64); pred[:, 0, :] = y1
    h = y1
    tn = M.times_t.cpu().numpy()
    for i in range(1, T):
        dt = max(tn[i] - tn[i - 1], 1e-6)
        h = h + np.einsum('bp,bqp->bq', h, K[ids]) * dt
        pred[:, i, :] = h
    return pred


def pcc_mse_from_pred(pred, Y_true):
    """Snapshot PCC / MSE of a (already ensembled) trajectory, t0 excluded."""
    pccs = []
    for t in range(1, T):
        rs = []
        for p in range(P):
            if np.std(Y_true[:, t, p]) < 1e-12: continue
            rs.append(pearsonr(pred[:, t, p], Y_true[:, t, p])[0])
        pccs.append(float(np.mean(rs)) if rs else float('nan'))
    mse = float(np.mean((pred[:, 1:, :] - Y_true[:, 1:, :]) ** 2))
    return float(np.nanmean(pccs)), mse


def gblup_extrap(Umat, in_tr, in_val, grm, lam=1.0):
    E = Umat.shape[1]
    ytr = Umat[in_tr] - Umat[in_tr].mean(0)
    Ktt = grm[np.ix_(in_tr, in_tr)]; Kte = grm[np.ix_(in_val, in_tr)]
    coef = np.linalg.solve(Ktt + lam * np.eye(len(in_tr)), ytr)
    pred = Kte @ coef
    out = np.full(E, np.nan)
    for j in range(E):
        if np.std(ytr[:, j]) < 1e-12 or np.std(Umat[in_val, j]) < 1e-12 or np.std(pred[:, j]) < 1e-12:
            continue
        out[j] = pearsonr(Umat[in_val, j], pred[:, j])[0]
    return out


def entry_stats(U, V, grm, in_tr, in_val):
    Ur = U.reshape(N, -1).astype(np.float64); Vr = V.reshape(N, -1).astype(np.float64)
    A = np.einsum("npq,nqr->npr", U, V).reshape(N, -1).astype(np.float64)   # A=U·V 逐元素(可识别)
    r = U.shape[2]
    sig = np.zeros((N, r), dtype=np.float64)                                 # A 的奇异值(可识别)
    for i in range(N):
        s = np.linalg.svd(U[i] @ V[i], compute_uv=False)
        sig[i] = s[:r]
    def h2(X): return float(np.nanmean(M.greml_h2_batch(X, grm)))
    def cv(X): return float(np.nanmean(M.gblup_cv_centered_batch(X, grm)))
    def ext(X): return float(np.nanmean(gblup_extrap(X, in_tr, in_val, grm)))
    return dict(h2_U=h2(Ur), h2_V=h2(Vr), cv_U=cv(Ur), cv_V=cv(Vr), ext_U=ext(Ur), ext_V=ext(Vr),
                h2_A=h2(A), cv_A=cv(A), ext_A=ext(A),
                h2_sig=h2(sig), cv_sig=cv(sig), ext_sig=ext(sig))


def run_fold(fold, ranks, conds, n_epochs):
    """跑单个 fold-set: 折内 5 折(4/5 训练 + 1/5 早停) -> 集成; 提取 U/V 算 h2/ext, 汇总 pcc/mse, 写 f{fold}.csv."""
    global P, T, N, noise2_all
    M.load_data(GRM_DIM, fold)
    P, T, N = M.P, M.T, M.N
    noise2_all = np.random.RandomState(7).randn(N, P).astype(np.float32)
    in_tr, in_val = M.fold_idx(fold)
    fc = M.cv_foldsets[in_tr, fold]                 # 折内标签 1..5 (fold6 已被 fold_idx 排除)
    Ys_val = ((M.Y_all[in_val] - M.tr_min) / M.tr_rng).astype(np.float32)
    csv_path = OUT / f'knode_rank_genetics_value_f{fold}.csv'
    rows = pd.read_csv(csv_path).to_dict('records') if csv_path.exists() else []
    energy_rows = []
    # (full, rank=2) 排到最前: 该格的 5 个 inner 模型创建顺序与主脚本逐次对齐 (RNG 流一致)
    if 'full' in conds and 2 in ranks:
        conds = ['full'] + [c for c in conds if c != 'full']
        ranks = [2] + [r for r in ranks if r != 2]
    for cond in conds:
        for rank in ranks:
            preds, stats = [], []
            for inner in range(1, 6):
                itr = in_tr[fc != inner]; iva = in_tr[fc == inner]
                Ys_in = ((M.Y_all[itr] - M.tr_min) / M.tr_rng).astype(np.float32)
                Ys_ea = ((M.Y_all[iva] - M.tr_min) / M.tr_rng).astype(np.float32)
                f = UV2 / f'gen_{cond}_r{rank}_f{fold}_i{inner}.npz'
                if f.exists():
                    z = np.load(f); U = z['U']; V = z['V']; pv = z['pred']
                else:
                    t0 = time.time()
                    m = train_model(cond, rank, Ys_in, itr, Ys_ea, iva, n_epochs)
                    U, V = extract_uv(m, cond)
                    pv = predict_torch(m, cond, in_val)
                    np.savez_compressed(f, U=U.astype(np.float32), V=V.astype(np.float32),
                                        pred=pv.astype(np.float32))
                    print(f'  [train {cond}/r{rank}/inner{inner}] {time.time()-t0:.1f}s', flush=True)
                    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                preds.append(pv)
                if rank == 50:
                    ce = cum_energy((U @ V).astype(np.float32)).mean(0)
                    for k in [1, 2, 3, 4, 6, 8, 10, 16, 20, 30, 50]:
                        energy_rows.append({'cond': cond, 'k': k, 'energy': round(float(ce[k - 1]), 6)})
                stats.append(entry_stats(U, V, M.grm_raw, in_tr, in_val))
            pcc, mse = pcc_mse_from_pred(np.mean(np.stack(preds), axis=0), Ys_val)
            st = {k: float(np.nanmean([s[k] for s in stats])) for k in stats[0]}
            if cond == 'const':
                st['h2_U'] = 0.0; st['h2_V'] = 0.0; st['h2_A'] = 0.0; st['h2_sig'] = 0.0
            rows = [r for r in rows if not (r['cond'] == cond and r['rank'] == rank)]
            rows.append({'cond': cond, 'rank': rank,
                         'h2_U': round(st['h2_U'], 4), 'h2_V': round(st['h2_V'], 4),
                         'cv_U': round(st['cv_U'], 4), 'cv_V': round(st['cv_V'], 4),
                         'ext_U': round(st['ext_U'], 4), 'ext_V': round(st['ext_V'], 4),
                         'h2_A': round(st['h2_A'], 4), 'cv_A': round(st['cv_A'], 4), 'ext_A': round(st['ext_A'], 4),
                         'h2_sig': round(st['h2_sig'], 4), 'cv_sig': round(st['cv_sig'], 4), 'ext_sig': round(st['ext_sig'], 4),
                         'pcc': round(pcc, 4), 'mse': round(mse, 6)})
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f'  [{cond}/r{rank}] h2_U={st["h2_U"]:.3f} ext_U={st["ext_U"]:.3f} pcc={pcc:.4f}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f'\nSaved -> {csv_path}')
    if energy_rows:
        ep = OUT / f'knode_rank_energy_f{fold}.csv'
        newe = pd.DataFrame(energy_rows)
        if ep.exists():
            olde = pd.read_csv(ep)
            keep = olde[~olde.set_index(['cond', 'k']).index.isin(newe.set_index(['cond', 'k']).index)]
            newe = pd.concat([keep, newe], ignore_index=True)
        newe.to_csv(ep, index=False)
        print(f'Saved -> {ep}')


def aggregate():
    """把 knode_rank_genetics_value_f{0..9}.csv 聚合成 10fold_agg.csv（mean±std）。"""
    gf = sorted(OUT.glob('knode_rank_genetics_value_f[0-9].csv'))
    if not gf:
        print('no per-fold CSV found'); return
    gdf = pd.concat([pd.read_csv(f) for f in gf])
    gagg = gdf.groupby(['cond', 'rank']).agg(
        h2_U_mean=('h2_U', 'mean'), h2_U_std=('h2_U', 'std'),
        h2_V_mean=('h2_V', 'mean'), h2_V_std=('h2_V', 'std'),
        cv_U_mean=('cv_U', 'mean'), cv_V_mean=('cv_V', 'mean'),
        ext_U_mean=('ext_U', 'mean'), ext_U_std=('ext_U', 'std'),
        ext_V_mean=('ext_V', 'mean'), ext_V_std=('ext_V', 'std'),
        h2_A_mean=('h2_A', 'mean'), h2_A_std=('h2_A', 'std'),
        cv_A_mean=('cv_A', 'mean'), ext_A_mean=('ext_A', 'mean'), ext_A_std=('ext_A', 'std'),
        h2_sig_mean=('h2_sig', 'mean'), h2_sig_std=('h2_sig', 'std'),
        cv_sig_mean=('cv_sig', 'mean'), ext_sig_mean=('ext_sig', 'mean'), ext_sig_std=('ext_sig', 'std'),
        pcc_mean=('pcc', 'mean'), pcc_std=('pcc', 'std'),
        mse_mean=('mse', 'mean'), mse_std=('mse', 'std'),
        n=('pcc', 'count'),
    ).reset_index()
    # const（共享算子，无遗传输入）h2 语义为 0：既有逐折 CSV 里若仍为 GREML 近常数列伪影 1.0，此处统一置 0
    gagg.loc[gagg['cond'] == 'const', ['h2_U_mean', 'h2_U_std', 'h2_V_mean', 'h2_V_std', 'h2_A_mean', 'h2_A_std', 'h2_sig_mean', 'h2_sig_std']] = 0.0
    gagg.to_csv(OUT / 'knode_rank_genetics_value_10fold_agg.csv', index=False)
    print(f'\nAggregated {len(gf)} folds -> knode_rank_genetics_value_10fold_agg.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=str, default='0,1,2,3,4,5,6,7,8,9',
                    help='逗号分隔的 fold-set 列表，如 "0,1,2"')
    ap.add_argument('--n_epochs', type=int, default=1000)
    ap.add_argument('--ranks', type=str, default=None)
    ap.add_argument('--conds', type=str, default=None)
    ap.add_argument('--aggregate', action='store_true',
                    help='仅聚合已有 knode_rank_genetics_value_f{fold}.csv')
    args = ap.parse_args()

    ranks = [int(x) for x in args.ranks.split(',')] if args.ranks else RANKS
    conds = args.conds.split(',') if args.conds else CONDS

    if args.aggregate:
        aggregate()
        return

    folds = [int(x) for x in args.folds.split(',') if x.strip()]
    print(f'Device={DEVICE} | folds={folds} | ranks={ranks} | conds={conds}', flush=True)
    for fold in folds:
        print(f'\n===== fold {fold} =====', flush=True)
        run_fold(fold, ranks, conds, args.n_epochs)
    aggregate()


if __name__ == '__main__':
    main()
