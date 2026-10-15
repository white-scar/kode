#!/usr/bin/env python3
"""knode_rank_select_v2_affine.py -- rank selection on the FULL KNode (affine).

Based on knode_rank_select_v2.py, but the model is replaced by the full KNode:
  drift : dh/dt = h.(U V)^T + c_i    (low-rank operator + genotype-specific offset c_i)
The rank analysis (rank of the operator U,V) is unchanged; c_i is always ON (not ablated).
Output naming: rank_select_v2_affine{tag}_f{fold}.csv and
rank_mode_heritability_v2_affine{tag}_f{fold}.csv

Usage: python knode_rank_select_v2_affine.py --fold 0 --n_epochs 1000 [--stage both] [--max_rank 3]
"""
import math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr
from pathlib import Path
import argparse, warnings, time
warnings.filterwarnings('ignore')

import knode_perfect_vs_predicted_operators as M

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 1e-3; WEIGHT_DECAY = 1e-4; HID = 128
GRM_DIM = 50
CONDS = ['full', 'grmonly', 'tp1only', 'noise']
SPEC_RANKS = [2, 3, 4, 6, 8, 10, 16, 20, 30, 50]
ENERGY_CUT = 0.90
OUT_DIR = Path('../outputs/reproduce'); OUT_DIR.mkdir(parents=True, exist_ok=True)

P = T = N = None
noise2_all = None


# ---------------- full KNode model (rank-r operator + c_i) ----------------
class RankAffineModel(nn.Module):
    def __init__(self, P, rank, hid=128):
        super().__init__()
        self.P = P; self.rank = rank
        self.cond_in = GRM_DIM + P
        self.hyper_enc = nn.Sequential(nn.Linear(self.cond_in, hid), nn.Tanh())
        self.hyper_U = nn.Linear(hid, P * rank)
        self.hyper_V = nn.Linear(hid, rank * P)
        self.hyper_b = nn.Linear(hid, P)          # c_i (genotype-specific offset)
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
    if cond == 'full':
        g = torch.tensor(M.grm_pca[ids], device=DEVICE); t = torch.tensor(M.y1_all[ids], device=DEVICE)
    elif cond == 'grmonly':
        g = torch.tensor(M.grm_pca[ids], device=DEVICE); t = None
    elif cond == 'tp1only':
        g = None; t = torch.tensor(M.y1_all[ids], device=DEVICE)
    elif cond == 'noise':
        g = torch.tensor(M.noise_all[ids], device=DEVICE); t = torch.tensor(noise2_all[ids], device=DEVICE)
    else:
        raise ValueError(cond)
    return g, t


def train_model(cond, rank, Ys_tr, train_ids, Ys_early, early_ids, seed, n_epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    m = RankAffineModel(P, rank, HID).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    y_tr = torch.tensor(Ys_tr.astype(np.float32), device=DEVICE)
    g_tr, t_tr = channels(cond, train_ids)
    y_early = torch.tensor(Ys_early.astype(np.float32), device=DEVICE)
    g_early, t_early = channels(cond, early_ids)
    best = None; best_loss = float('inf')
    for ep in range(n_epochs):
        m.train(); m.set_z(g_tr, t_tr)
        pred = euler_integrate(m, y_tr[:, 0, :], M.times_t)
        loss = F.mse_loss(pred[:, 1:, :], y_tr[:, 1:, :])
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


def extract_full(m, cond):
    """Extract operator U,V and the offset c_i for all N genotypes."""
    ids = np.arange(N)
    g, t = channels(cond, ids)
    with torch.no_grad():
        m.set_z(g, t)
        U = m._U_dyn.cpu().numpy().copy()
        V = m._V_dyn.cpu().numpy().copy()
        B = m._b_dyn.cpu().numpy().copy()
    return U, V, B


def roll_pcc_mse(K, ids, Y_true, B=None):
    y1 = M.y1_all[ids]
    pred = np.zeros((len(ids), T, P), dtype=np.float64); pred[:, 0, :] = y1
    h = y1
    tn = M.times_t.cpu().numpy()
    for i in range(1, T):
        dt = max(tn[i] - tn[i - 1], 1e-6)
        drift = np.einsum('bp,bqp->bq', h, K[ids])
        if B is not None:
            drift = drift + B[ids]
        h = h + drift * dt
        pred[:, i, :] = h
    pccs = []
    for t in range(1, T):
        rs = []
        for p in range(P):
            if np.std(Y_true[:, t, p]) < 1e-12: continue
            rs.append(pearsonr(pred[:, t, p], Y_true[:, t, p])[0])
        pccs.append(float(np.mean(rs)) if rs else float('nan'))
    mse = float(np.mean((pred[:, 1:, :] - Y_true[:, 1:, :]) ** 2))
    return float(np.nanmean(pccs)), mse


def cum_energy(K):
    sv = np.linalg.svd(K, compute_uv=False)
    esq = sv ** 2
    return np.cumsum(esq, axis=1) / esq.sum(1)[:, None]


def mode_heritability(K, grm):
    n_, P_, _ = K.shape
    S = np.zeros((n_, P_)); Ua = np.zeros((n_, P_, P_))
    for g in range(n_):
        u, s, vt = np.linalg.svd(K[g])
        S[g] = s
        for j in range(P_):
            if u[0, j] < 0:
                u[:, j] *= -1
        Ua[g] = u
    h2_s = M.greml_h2_batch(S, grm)
    acc_s = M.gblup_cv_centered_batch(S, grm)
    h2_U = np.zeros(P_); acc_U = np.zeros(P_)
    for j in range(P_):
        h2_U[j] = float(np.nanmean(M.greml_h2_batch(Ua[:, :, j], grm)))
        acc_U[j] = float(np.nanmean(M.gblup_cv_centered_batch(Ua[:, :, j], grm)))
    return S.mean(0), h2_s, acc_s, h2_U, acc_U


def main():
    global P, T, N, noise2_all
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--n_epochs', type=int, default=1000)
    ap.add_argument('--stage', type=str, default='predicted', choices=['predicted', 'perfect', 'both'])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--max_rank', type=int, default=3)
    args = ap.parse_args()

    M.load_data(GRM_DIM, args.fold)
    P, T, N = M.P, M.T, M.N
    noise2_all = np.random.RandomState(7).randn(N, P).astype(np.float32)
    print(f'Device={DEVICE} | N={N} P={P} T={T}', flush=True)

    if args.smoke:
        ranks = [2, 50]; n_epochs = 120; stages = ['predicted']; conds = CONDS; tag = '_smoke'
    else:
        ranks = sorted(set(SPEC_RANKS) | set(range(1, args.max_rank + 1)))
        n_epochs = args.n_epochs
        stages = ['perfect', 'predicted'] if args.stage == 'both' else [args.stage]
        conds = CONDS; tag = ''

    in_tr, in_val = M.fold_idx(args.fold)
    _fc_tr = M.cv_foldsets[in_tr, args.fold]
    in_train = in_tr[_fc_tr <= 4]   # fold1-4 训练
    in_early = in_tr[_fc_tr == 5]   # fold5 作为 inner 验证(early-stop)
    Ys_val = ((M.Y_all[in_val] - M.tr_min) / M.tr_rng).astype(np.float32)

    rows = []; mode_rows = []
    for stage in stages:
        for cond in conds:
            for rank in ranks:
                train_ids, early_ids = (in_val, in_val) if stage == 'perfect' else (in_train, in_early)
                Ys_tr = ((M.Y_all[train_ids] - M.tr_min) / M.tr_rng).astype(np.float32)
                Ys_early = ((M.Y_all[early_ids] - M.tr_min) / M.tr_rng).astype(np.float32)
                t0 = time.time()
                m = train_model(cond, rank, Ys_tr, train_ids, Ys_early, early_ids, 42, n_epochs)
                U, V, B = extract_full(m, cond)
                K = (U @ V).astype(np.float32)
                pcc, mse = roll_pcc_mse(K, in_val, Ys_val, B)
                row = {'stage': stage, 'cond': cond, 'rank': rank,
                       'pcc': round(pcc, 4), 'mse': round(mse, 6)}
                if rank == 50:
                    mc = cum_energy(K).mean(0)
                    k90 = int(np.argmax(mc >= ENERGY_CUT)) + 1
                    if mc[-1] < ENERGY_CUT: k90 = P
                    row['ceil_90pct'] = k90
                    for k in [1, 2, 3, 4, 5, 8, 11]:
                        if k <= P: row[f'en{k}'] = round(float(mc[k - 1]), 4)
                    s_mean, h2_s, acc_s, h2_U, acc_U = mode_heritability(K, M.grm_raw)
                    for j in range(P):
                        mode_rows.append({'stage': stage, 'cond': cond, 'mode': j + 1,
                                          's_mean': round(float(s_mean[j]), 6),
                                          'h2_s': round(float(h2_s[j]), 4),
                                          'acc_s': round(float(acc_s[j]), 4),
                                          'h2_U': round(float(h2_U[j]), 4),
                                          'acc_U': round(float(acc_U[j]), 4)})
                rows.append(row)
                print(f'  [{stage}/{cond}/r{rank}] pcc={pcc:.4f} mse={mse:.6f} ({time.time()-t0:.1f}s)', flush=True)
                if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f'rank_select_v2_affine{tag}_f{args.fold}.csv', index=False)
    if mode_rows:
        pd.DataFrame(mode_rows).to_csv(OUT_DIR / f'rank_mode_heritability_v2_affine{tag}_f{args.fold}.csv', index=False)

    print('\n===== rank_select_v2_affine summary =====')
    print(df[['stage', 'cond', 'rank', 'pcc', 'mse']].to_string(index=False))
    if mode_rows:
        mdf = pd.DataFrame(mode_rows)
        piv = mdf[(mdf['mode'] <= 6)].pivot_table(index=['stage', 'cond'], columns='mode', values='h2_s', aggfunc='first')
        print('\n===== full-rank mode heritability h2_s (first 6 modes) =====')
        print(piv.to_string())
    print('\nDone.')


if __name__ == '__main__':
    main()
