#!/usr/bin/env python3
"""knode_rice_np_tp1_affine.py -- Kode (KNode) main method, rice indica panel (control).

Model: hypernetwork encodes cond=[grm_pca, y1] -> (U_i, V_i, b_i) i.e. an operator A_i; state is
propagated by continuous-time explicit Euler integration (dt = normalized inter-observation step).
Two modes:
  ode_hyper  : recursive  -- operator generated from [grm, t1], rolled forward; no later truth used
  iter_hyper : iterative  -- same operator, but each step is teacher-forced with the true state

Protocol: 10 fold-sets; folds 1-5 train (inner-5: 4/5 train + 1/5 early stop), fold 6 test;
scaler fitted on the training folds only (per-trait min-max), shared with val/test.
Outputs: {PREFIX}{mode}{tag}_per_trait.csv / {PREFIX}all_detail{tag}.csv / {PREFIX}all_summary{tag}.csv
Usage: python knode_rice_np_tp1_affine.py [--epochs 1000] [--foldsets 10] [--mode ...]
"""
import math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from pathlib import Path
import time, warnings, argparse; warnings.filterwarnings('ignore')

torch.manual_seed(42); np.random.seed(42)
SEED = 42

DATA_DIR = Path('../data/rice_salinity')
OUT_DIR  = Path('../outputs/kode_result/rice_indica_control')
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

N_FOLDSETS = 1
N_EPOCHS = 1000
LR = 0.001
WEIGHT_DECAY = 1e-4
GRM_DIM = 10          # 2026-09-19 锁定 D_G=10 (原 20)
RANK = 2
HID = 128
MODES = ['ode_hyper', 'iter_hyper']


# ---------------- data load (identical to hyper-ode.py) ----------------
def load_data(n_foldsets):
    pheno = pd.read_csv(DATA_DIR / 'rice_indica_control_traits.csv', index_col=0)
    pheno = pheno[pheno['DAS'] >= 30].copy()   # 2026-09-20 对齐作者 dynamicGP 时间窗: 只用 DAS 30-42 (T=13), 丢掉 DAS 28
    cv_raw = pd.read_csv(DATA_DIR / 'rice_cv_10fold.csv', index_col=0)
    grm = pd.read_csv(DATA_DIR / 'rice_GRM_226x226.csv', index_col=0)
    pheno['bio_ID'] = pheno['bio_ID'].astype(str).str.strip()
    cv_raw.index = cv_raw.index.astype(str).str.strip()
    grm.index = grm.index.astype(str).str.strip()
    grm.columns = grm.columns.astype(str).str.strip()
    trait_cols = [c for c in pheno.columns if c not in ('bio_ID', 'accession', 'DAS')]
    P = len(trait_cols)
    times_all = np.array(sorted(pheno['DAS'].unique()), dtype=np.float64)
    T = len(times_all)
    common_ids = sorted(set(pheno['bio_ID'].astype(str)), key=int)
    N = len(common_ids)
    cv_foldsets = cv_raw.loc[common_ids].iloc[:, :n_foldsets].values.astype(int)
    grm = grm.loc[common_ids, common_ids].values.astype(np.float64)
    pca = PCA(n_components=GRM_DIM)
    grm_pca = pca.fit_transform(grm).astype(np.float32)
    Y_all = np.full((N, T, P), np.nan, dtype=np.float64)
    for pi, trait in enumerate(trait_cols):
        sub = pheno[['bio_ID', 'DAS', trait]].copy()
        sub['bio_ID'] = sub['bio_ID'].astype(str)
        Y_all[:, :, pi] = sub.pivot(index='bio_ID', columns='DAS', values=trait).loc[common_ids].values
    assert not np.any(np.isnan(Y_all))
    tn = (times_all - times_all.min()) / (times_all.max() - times_all.min() + 1e-8)
    return dict(P=P, T=T, N=N, trait_cols=trait_cols, times_all=times_all, tn=tn,
                grm_pca=grm_pca, Y_all=Y_all, cv_foldsets=cv_foldsets)


# ---------------- hyper ODE ----------------
class HyperODEModel(nn.Module):
    """低秩 Koopman ODE (U,V 由 MLP 超网络从 cond=[grm_pca,y1] 生成).
    dh/dt = h·Vᵀ·Uᵀ;  pred = ODE 积分 (自 h0 起自由展开)."""
    def __init__(self, P=50, COND_IN=100, rank=2, hid=128, drift='linear'):
        super().__init__()
        self.P = P; self.drift = drift
        self.hyper_enc = nn.Sequential(nn.Linear(COND_IN, hid), nn.Tanh())
        self.hyper_U = nn.Linear(hid, P * rank)
        self.hyper_V = nn.Linear(hid, rank * P)
        if drift == 'affine':
            self.hyper_b = nn.Linear(hid, P)
        self._U_dyn = self._V_dyn = self._b_dyn = None

    def set_z(self, grm_pca, y1):
        cond = torch.cat([grm_pca, y1], dim=-1)
        B = grm_pca.shape[0]
        h = self.hyper_enc(cond)
        self._U_dyn = self.hyper_U(h).view(B, self.P, RANK)
        self._V_dyn = self.hyper_V(h).view(B, RANK, self.P)
        if self.drift == 'affine':
            self._b_dyn = self.hyper_b(h).view(B, self.P)

    def forward(self, t, h):
        lin = torch.bmm(
            torch.bmm(h.unsqueeze(1), self._V_dyn.transpose(1, 2)),
            self._U_dyn.transpose(1, 2)
        ).squeeze(1)
        return lin + self._b_dyn if self.drift == 'affine' else lin

def euler_integrate(model, h0, ts):
    B, D = h0.shape; T_len = len(ts)
    traj = torch.zeros(B, T_len, D, device=h0.device, dtype=h0.dtype); traj[:, 0] = h0
    h = h0
    for i in range(1, T_len):
        dt = max(ts[i].item() - ts[i - 1].item(), 1e-6)
        h = h + model(None, h) * dt; traj[:, i] = h
    return traj


def ode_forward(ode, y_input, times_t, is_iter):
    if is_iter:
        B, T_len, Pn = y_input.shape
        out = torch.zeros_like(y_input)
        out[:, 0, :] = y_input[:, 0, :]
        for t in range(T_len - 1):
            h_t = y_input[:, t, :]
            dt = max(times_t[t + 1].item() - times_t[t].item(), 1e-6)
            out[:, t + 1, :] = h_t + ode(None, h_t) * dt
        return out
    else:
        h0 = y_input[:, 0, :]
        return euler_integrate(ode, h0, times_t)


# ---------------- metrics (identical to hyper-ode.py, dGP 口径) ----------------
def compute_metrics(pred, true, times, trait_names=None, true_norm=None, pred_norm=None):
    n_geno, T_len, n_p = pred.shape
    pcc_tp = {t: [] for t in times}; r2_tp = {t: [] for t in times}
    nrmse_tp = {t: [] for t in times}; mse_tp = {t: [] for t in times}
    mse_norm_tp = {t: [] for t in times}
    r_long_all = []; trait_rows = []
    has_norm = (true_norm is not None) and (pred_norm is not None)
    for tpi, tp in enumerate(times):
        for pi in range(n_p):
            yp = pred[:, tpi, pi]; yt = true[:, tpi, pi]
            mask = ~np.isnan(yp) & ~np.isnan(yt)
            if mask.sum() < 3: continue
            r, _ = pearsonr(yt[mask], yp[mask])
            if np.isnan(r): continue
            pcc_tp[tp].append(r)
            ss_res = np.sum((yt[mask] - yp[mask]) ** 2)
            ss_tot = np.sum((yt[mask] - np.mean(yt[mask])) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan; r2_tp[tp].append(r2)
            mse = np.mean((yp[mask] - yt[mask]) ** 2); mse_tp[tp].append(mse)
            rmse = np.sqrt(mse); denom = abs(np.mean(yt[mask]))
            nr = rmse / denom if denom > 1e-8 else np.nan; nrmse_tp[tp].append(nr)
            mse_n = float(np.nan)
            if has_norm:
                pn = pred_norm[:, tpi, pi]; tn_ = true_norm[:, tpi, pi]
                mse_n = float(np.mean((tn_[mask] - pn[mask]) ** 2))
            mse_norm_tp[tp].append(mse_n)
            trait_rows.append({'timepoint': int(tp), 'trait_idx': pi,
                'trait': trait_names[pi] if trait_names else str(pi),
                'pcc': float(r), 'R2': float(r2) if not np.isnan(r2) else float('nan'),
                'NRMSE': float(nr) if not np.isnan(nr) else float('nan'),
                'MSE_raw': float(mse), 'MSE_norm': mse_n})
    for gi in range(n_geno):
        pv = pred[gi].ravel(); tv = true[gi].ravel()
        mask = ~np.isnan(pv) & ~np.isnan(tv)
        if mask.sum() >= 3:
            r, _ = pearsonr(tv[mask], pv[mask])
            if not np.isnan(r): r_long_all.append(r)
    return pcc_tp, r2_tp, nrmse_tp, mse_tp, mse_norm_tp, r_long_all, trait_rows


# ---------------- training ----------------
def train_hyper(Y_tr, Y_val, grm_tr, grm_val, times_t, is_iter, epochs, drift='linear'):
    n_tr, Tn, Pn = Y_tr.shape
    ode = HyperODEModel(Pn, GRM_DIM + Pn, RANK, HID, drift).to(DEVICE)
    opt = torch.optim.Adam(ode.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    y_tr = torch.tensor(Y_tr.astype(np.float32), device=DEVICE)
    y_val = torch.tensor(Y_val.astype(np.float32), device=DEVICE)
    e_tr = torch.tensor(grm_tr, device=DEVICE); e_val = torch.tensor(grm_val, device=DEVICE)
    h0_tr = y_tr[:, 0, :]; h0_val = y_val[:, 0, :]
    best = float('inf'); best_state = None
    for ep in range(epochs):
        ode.train()
        ode.set_z(e_tr, h0_tr)
        ode_tr = ode_forward(ode, y_tr, times_t, is_iter)
        loss = F.mse_loss(ode_tr, y_tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ode.parameters(), 1.0)
        opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            with torch.no_grad():
                ode.eval()
                ode.set_z(e_val, h0_val)
                ov = ode_forward(ode, y_val, times_t, is_iter)
                vl = F.mse_loss(ov[:, 1:, :], y_val[:, 1:, :]).item()
            if best_state is None or (not math.isnan(vl) and vl < best):
                best = vl; best_state = {k: v.cpu().clone() for k, v in ode.state_dict().items()}
    ode.load_state_dict(best_state); ode.eval()
    return ode


def ode_predict(ode, grm, y_in, times_t, is_iter):
    with torch.no_grad():
        g = torch.tensor(np.asarray(grm, dtype=np.float32), device=DEVICE)
        y = torch.tensor(np.asarray(y_in, dtype=np.float32), device=DEVICE)
        ode.set_z(g, y[:, 0, :])
        out = ode_forward(ode, y, times_t, is_iter)
        return out.cpu().numpy()


# ---------------- main ----------------
def main():
    global N_EPOCHS, N_FOLDSETS, MODES, DRIFT, TAG, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=N_EPOCHS)
    ap.add_argument('--foldsets', type=int, default=N_FOLDSETS)
    ap.add_argument('--mode', type=str, default=None, help='只跑指定模式, 逗号分隔')
    ap.add_argument('--drift', type=str, default='affine', choices=['linear', 'affine'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=SEED)
    a, _ = ap.parse_known_args()
    N_EPOCHS = a.epochs; N_FOLDSETS = a.foldsets; DRIFT = a.drift; TAG = a.tag
    SEED = a.seed
    if a.mode: MODES = [m.strip() for m in a.mode.split(',')]

    D = load_data(N_FOLDSETS)
    P, T, N = D['P'], D['T'], D['N']
    trait_cols = D['trait_cols']; times_all = D['times_all']; tn = D['tn']
    grm_pca = D['grm_pca']; Y_all = D['Y_all']; cv_foldsets = D['cv_foldsets']
    keep_idx = list(range(T)); keep_times = times_all[keep_idx]
    times_t = torch.tensor(tn, dtype=torch.float32, device=DEVICE)

    print(f'[{DEVICE}] fold-sets={N_FOLDSETS} epochs={N_EPOCHS} modes={MODES}')

    all_summary = []; all_detail = []; rows_pf = []
    for mode in MODES:
        print(f'\n{"="*60}\n  mode: {mode}   (h0 = D{int(keep_times[0])})\n{"="*60}')
        is_iter = 'iter' in mode
        SPLITS = ['train', 'test']
        pcc_all = {s: {t: [] for t in keep_times} for s in SPLITS}
        r2_all = {s: {t: [] for t in keep_times} for s in SPLITS}
        nr_all = {s: {t: [] for t in keep_times} for s in SPLITS}
        ms_all = {s: {t: [] for t in keep_times} for s in SPLITS}
        msn_all = {s: {t: [] for t in keep_times} for s in SPLITS}
        rl_total = {s: [] for s in SPLITS}
        all_trait_rows = []
        t_start = time.time()

        for foldset in range(N_FOLDSETS):
            torch.manual_seed(SEED); np.random.seed(SEED)   # 2026-09-21: reset per foldset so foldset-f init == cond-ladder fold-f init
            fold_col = cv_foldsets[:, foldset]
            tr_mask = (fold_col >= 1) & (fold_col <= 5); te_mask = (fold_col == 6)
            tr_idx = np.where(tr_mask)[0]; te_idx = np.where(te_mask)[0]
            print(f'  Fold-set {foldset+1}/{N_FOLDSETS} (tr={len(tr_idx)}, te={len(te_idx)})...', flush=True)
            tr_fold_assign = fold_col[tr_idx]

            Y_tr_full = Y_all[tr_idx][:, keep_idx, :]
            tr_min = np.nanmin(Y_tr_full, axis=(0, 1)); tr_max = np.nanmax(Y_tr_full, axis=(0, 1))
            tr_rng = np.maximum(tr_max - tr_min, 1e-8)

            te_preds_accum = []; tr_preds_accum = []; val_pos_accum = []
            for inner_fold in range(1, 6):
                in_tr = (tr_fold_assign >= 1) & (tr_fold_assign <= 5) & (tr_fold_assign != inner_fold)
                in_val = (tr_fold_assign == inner_fold)
                in_tr_idx = tr_idx[in_tr]; in_val_idx = tr_idx[in_val]
                Y_in_tr = Y_all[in_tr_idx][:, keep_idx, :]
                Ys_in_tr = (Y_in_tr - tr_min) / tr_rng
                Ys_in_val = (Y_all[in_val_idx][:, keep_idx, :] - tr_min) / tr_rng
                Ys_te = (Y_all[te_idx][:, keep_idx, :] - tr_min) / tr_rng

                ode = train_hyper(Ys_in_tr, Ys_in_val, grm_pca[in_tr_idx], grm_pca[in_val_idx],
                                  times_t, is_iter, N_EPOCHS, DRIFT)
                ode_tr = ode_predict(ode, grm_pca[in_tr_idx], Ys_in_tr, times_t, is_iter)
                ode_val = ode_predict(ode, grm_pca[in_val_idx], Ys_in_val, times_t, is_iter)
                ode_te = ode_predict(ode, grm_pca[te_idx], Ys_te, times_t, is_iter)
                pred_val = ode_val * tr_rng + tr_min
                pred_te = ode_te * tr_rng + tr_min

                tr_preds_accum.append(pred_val)
                val_pos_accum.append(np.where(in_val)[0])
                te_preds_accum.append(pred_te)
                if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                print(f'    inner {inner_fold} done', end=' ', flush=True)
            print('', flush=True)

            pred_raw_te = np.mean(np.stack(te_preds_accum), axis=0)
            true_raw_te = Y_all[te_idx][:, keep_idx, :]
            pred_raw_te[:, 0, :] = true_raw_te[:, 0, :]
            true_raw_tr = Y_all[tr_idx][:, keep_idx, :]
            pred_raw_tr = np.zeros_like(true_raw_tr)
            for rel_pos, pv in zip(val_pos_accum, tr_preds_accum):
                pred_raw_tr[rel_pos] = pv
            pred_raw_tr[:, 0, :] = true_raw_tr[:, 0, :]

            true_norm_tr = (true_raw_tr - tr_min) / tr_rng
            pred_norm_tr = (pred_raw_tr - tr_min) / tr_rng
            pcc_tr, r2_tr, nr_tr, ms_tr, msn_tr, rl_tr, rows_tr = compute_metrics(
                pred_raw_tr, true_raw_tr, keep_times, trait_cols,
                true_norm=true_norm_tr, pred_norm=pred_norm_tr)
            for row in rows_tr: row['foldset'] = foldset + 1; row['mode'] = mode; row['split'] = 'train'
            all_trait_rows.extend(rows_tr)

            true_norm_te = (true_raw_te - tr_min) / tr_rng
            pred_norm_te = (pred_raw_te - tr_min) / tr_rng
            pcc_te, r2_te, nr_te, ms_te, msn_te, rl_te, rows_te = compute_metrics(
                pred_raw_te, true_raw_te, keep_times, trait_cols,
                true_norm=true_norm_te, pred_norm=pred_norm_te)
            for row in rows_te: row['foldset'] = foldset + 1; row['mode'] = mode; row['split'] = 'test'
            all_trait_rows.extend(rows_te)

            for split, (pcc, r2, nr, ms, msn, rl) in [('train', (pcc_tr, r2_tr, nr_tr, ms_tr, msn_tr, rl_tr)),
                                                      ('test', (pcc_te, r2_te, nr_te, ms_te, msn_te, rl_te))]:
                fold_vals = [v for tp, vals in pcc.items() if tp != keep_times[0] for v in vals if not np.isnan(v)]
                fr = float(np.mean(fold_vals)) if fold_vals else float('nan')
                print(f'    [{split}] pcc={fr:.4f}', flush=True)
                for tp in keep_times:
                    if tp == keep_times[0]: continue
                    pcc_all[split][tp].extend(pcc[tp]); r2_all[split][tp].extend(r2[tp])
                    nr_all[split][tp].extend(nr[tp]); ms_all[split][tp].extend(ms[tp])
                    msn_all[split][tp].extend(msn[tp])
                    rows_pf.append({'mode': mode, 'split': split, 'foldset': foldset + 1,
                                    'timepoint': int(tp),
                                    'pcc': float(np.mean(pcc[tp])),
                                    'MSE_norm': float(np.mean(msn[tp])),
                                    'MSE_raw': float(np.mean(ms[tp])),
                                    'NRMSE': float(np.mean(nr[tp]))})
                rl_total[split].extend(rl)
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()

        elapsed = round((time.time() - t_start) / 60, 1)
        for split in SPLITS:
            for tp in keep_times:
                if tp == keep_times[0]: continue
                all_detail.append({'mode': mode, 'split': split, 'timepoint': int(tp),
                    'pcc': float(np.mean(pcc_all[split][tp])), 'pcc_std': float(np.std(pcc_all[split][tp])),
                    'R2': float(np.mean(r2_all[split][tp])), 'R2_std': float(np.std(r2_all[split][tp])),
                    'MSE_raw': float(np.mean(ms_all[split][tp])), 'MSE_raw_std': float(np.std(ms_all[split][tp])),
                    'MSE_norm': float(np.mean(msn_all[split][tp])), 'MSE_norm_std': float(np.std(msn_all[split][tp])),
                    'NRMSE': float(np.mean(nr_all[split][tp])), 'NRMSE_std': float(np.std(nr_all[split][tp]))})
            av_pcc = [v for tp, v in pcc_all[split].items() if tp != keep_times[0] and v]
            mpcc = float(np.mean([np.mean(v) for v in av_pcc]))
            av_r2 = [v for tp, v in r2_all[split].items() if tp != keep_times[0] and v]
            mr2med = float(np.median([np.median(v) for v in av_r2])) if av_r2 else float('nan')
            av_msn = [v for tp, v in msn_all[split].items() if tp != keep_times[0] and v]
            mmsn = float(np.mean([np.mean(v) for v in av_msn])) if av_msn else float('nan')
            av_nr = [v for tp, v in nr_all[split].items() if tp != keep_times[0] and v]
            mnr = float(np.mean([np.mean(v) for v in av_nr])) if av_nr else float('nan')
            mrl = float(np.mean(rl_total[split])) if rl_total[split] else float('nan')
            print(f'\n  === {mode} [{split}] === pcc={mpcc:.4f} R2m={mr2med:.2f} '
                  f'MSE_norm={mmsn:.6f} NRMSE={mnr:.4f} r_long={mrl:.4f} [{elapsed} min]')
            all_summary.append({'mode': mode, 'split': split, 'pcc': mpcc, 'R2_median': mr2med,
                                'MSE_norm': mmsn, 'NRMSE': mnr, 'r_long': mrl,
                                'elapsed_min': elapsed, 'foldsets': N_FOLDSETS, 'epochs': N_EPOCHS})

        if all_trait_rows:
            pd.DataFrame(all_trait_rows).to_csv(OUT_DIR / f'knode_rice_tp1_{mode}{TAG}_per_trait.csv', index=False)

    pd.DataFrame(all_detail).to_csv(OUT_DIR / f'knode_rice_tp1_all_detail{TAG}.csv', index=False)
    pd.DataFrame(all_summary).to_csv(OUT_DIR / f'knode_rice_tp1_all_summary{TAG}.csv', index=False)
    if rows_pf:
        pd.DataFrame(rows_pf).to_csv(OUT_DIR / f'knode_rice_tp1_per_foldset{TAG}.csv', index=False)

    print('\n' + '=' * 60)
    print('SUMMARY (hyper ODE)')
    print('=' * 60)
    print(f'{"Mode":>16s} {"Split":>5s} {"PCC":>8s} {"R2m":>8s} {"MSE_norm":>10s} {"NRMSE":>7s}')
    for row in all_summary:
        print(f'{row["mode"]:>16s} {row["split"]:>5s} {row["pcc"]:8.4f} {row["R2_median"]:8.2f} '
              f'{row["MSE_norm"]:10.6f} {row["NRMSE"]:7.4f}')
    print('\nSaved -> outputs/reproduce/knode_rice_tp1_all_summary.csv / _all_detail.csv')


if __name__ == '__main__':
    main()
