#!/usr/bin/env python3
"""knode_hyper_ode_arab_affine.py — 拟南芥 hyper 变体 (affine 漂移) (MLP 超网络生成 U,V) KNode 论文口径对比.

参考玉米 reproduce/hyper-ode.py, 数据换成拟南芥 sg 接口 (knode_arab_data.py).
协议 (用户 2026-08-19 指定, 与动态基线可比):
  - 对齐作者 CV: 外部 5:1 (folds1-5 train, fold6 test); train 内 CV_looper 5 折循环 (4 tr/1 val)
  - 集成: test 预测 = 5 个 inner 模型均值 (不用 best_fold 选择)
  - 归一化: 训练 (含 inner val) 用训练集自身 min/max, 测试用验证集自身 min/max, 各自独立
  - MSE 在归一化空间计算 (MSE_norm); 同时输出 NRMSE / R2 / r_long; 剔 t0 (pcc=1 点)
变体:
  ode_hyper  : recursive ODE (从 h0 一次性积分整条轨迹), U,V = MLP(cond)
  iter_hyper : iterative ODE (逐时点 teacher-force), U,V = MLP(cond)
cond = [GRM_PC, TP1], GRM 维数默认 sg.GRM_DIM=40 (拟南芥维度扫描最优, 与动态基线对齐).
输出: outputs/reproduce/knode_hyper_ode_arab_all_summary{tag}.csv / _all_detail{tag}.csv / _per_trait{tag}.csv
用法 (在 reproduce/ 目录): python knode_hyper_ode_arab.py [--rank 2] [--grm_dim 40] [--tag ]
"""
import math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr
from pathlib import Path
import time, warnings, argparse; warnings.filterwarnings('ignore')

import knode_arab_data as sg  # 拟南芥数据模块 (get_S/build_cond/euler_integrate/times_t)

torch.manual_seed(42); np.random.seed(42)

OUT_DIR = sg.OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = sg.DEVICE

N_FOLDSETS = 10
N_EPOCHS = 1000
LR = sg.LR
WEIGHT_DECAY = sg.WEIGHT_DECAY
RANK = 2
HID = 128                  # hyper 隐层宽度 (同玉米 hyper-ode.py)

MODES = ['ode_hyper', 'iter_hyper']

_a0 = argparse.ArgumentParser(add_help=False)
_a0.add_argument('--rank', type=int, default=RANK)
_a0.add_argument('--grm_dim', type=int, default=sg.GRM_DIM)
_a0.add_argument('--drift', type=str, default='affine', choices=['linear', 'affine'])
_a0.add_argument('--tag', default='')
_a0, _ = _a0.parse_known_args()
RANK = _a0.rank; GRM_DIM = _a0.grm_dim; DRIFT = _a0.drift; _TAG = _a0.tag
print(f'  [--rank]->{RANK}  [--grm_dim]->{GRM_DIM}  [--tag]->{_TAG!r}', flush=True)

# ---- data (from sg, GRM 取前 GRM_DIM 个 PC) ----
print('=' * 60); print('KNode Hyper (Arabidopsis) — Loading data'); print('=' * 60)
S_all = sg.get_S(GRM_DIM).astype(np.float32)          # (N, GRM_DIM)
Y_all = sg.Y_all
times_all = sg.times_all; times_t = sg.times_t
T, P = Y_all.shape[1], Y_all.shape[2]
trait_cols = sg.trait_cols
common_ids = sg.common_ids
cv_foldsets = sg.cv_raw.loc[common_ids].iloc[:, :N_FOLDSETS].values   # (N, N_FOLDSETS), 1-5 train / 6 test
N = len(common_ids)
print(f'  Genotypes: {N} | Traits: {P} | Timepoints: {T} | GRM PCA dim: {GRM_DIM}', flush=True)


# ---- hyper ODE: cond -> MLP -> U,V (recursive/iterative 共用) ----
class HyperODEModel(nn.Module):
    """低秩 Koopman ODE, U,V 由 MLP 超网络从 cond=[grm_pca, y1] 生成.
    U: (B,P,RANK), V: (B,RANK,P), dh/dt = h·V·U  (与玉米 hyper-ode.py 逐位一致)."""
    def __init__(self, P=50, COND_IN=100, rank=1, hid=128, drift='affine'):
        super().__init__()
        self.P = P
        self.hyper_enc = nn.Sequential(nn.Linear(COND_IN, hid), nn.Tanh())
        self.hyper_U = nn.Linear(hid, P * rank)
        self.hyper_V = nn.Linear(hid, rank * P)
        self.drift = drift
        self._U_dyn = self._V_dyn = None
        self._b_dyn = None
        if drift == 'affine':
            self.hyper_b = nn.Linear(hid, P)

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


# ---- metrics (dGP 口径, 同玉米) ----
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


# ---- training: recursive / iterative hyper ODE (MSE on normalized space) ----
def train_hyper(Y_tr, Y_val, grm_tr, grm_val, times_t, is_iter):
    n_tr, Tn, Pn = Y_tr.shape
    ode = HyperODEModel(Pn, GRM_DIM + Pn, RANK, HID, DRIFT).to(DEVICE)
    opt = torch.optim.Adam(ode.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    y_tr = torch.tensor(Y_tr.astype(np.float32), device=DEVICE)
    y_val = torch.tensor(Y_val.astype(np.float32), device=DEVICE)
    e_tr = torch.tensor(grm_tr, device=DEVICE); e_val = torch.tensor(grm_val, device=DEVICE)
    best = float('inf'); best_state = None
    for ep in range(N_EPOCHS):
        ode.train()
        ode.set_z(e_tr, y_tr[:, 0, :])
        ode_tr = ode_forward(ode, y_tr, times_t, is_iter)
        loss = F.mse_loss(ode_tr, y_tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ode.parameters(), 1.0)
        opt.step()
        if ep % 50 == 0:
            with torch.no_grad():
                ode.eval()
                ode.set_z(e_val, y_val[:, 0, :])
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
        return ode_forward(ode, y, times_t, is_iter).cpu().numpy()


# ---- main ----
def main():
    global N_EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--foldsets', type=int, default=N_FOLDSETS)
    a, _ = ap.parse_known_args()
    N_EPOCHS = a.epochs
    n_foldsets = a.foldsets

    keep_idx = list(range(T)); keep_times = times_all[keep_idx]; n_tp = len(keep_times)

    all_summary = []; all_detail = []

    for mode in MODES:
        print(f'\n[{"="*56}\n  Ablation: {mode}  (h0 = D{int(keep_times[0])})\n{"="*56}', flush=True)
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

        for foldset in range(n_foldsets):
            fold_col = cv_foldsets[:, foldset]
            tr_mask = (fold_col >= 1) & (fold_col <= 5); te_mask = (fold_col == 6)
            tr_idx = np.where(tr_mask)[0]; te_idx = np.where(te_mask)[0]
            print(f'  Fold-set {foldset+1:2d}/{n_foldsets} (tr={len(tr_idx)}, te={len(te_idx)})...', flush=True)
            tr_fold_assign = fold_col[tr_idx]

            Y_tr_full = Y_all[tr_idx][:, keep_idx, :]
            tr_min = np.nanmin(Y_tr_full, axis=(0, 1)); tr_max = np.nanmax(Y_tr_full, axis=(0, 1))
            tr_rng = np.maximum(tr_max - tr_min, 1e-8)
            Y_te_full = Y_all[te_idx][:, keep_idx, :]
            te_min = np.nanmin(Y_te_full, axis=(0, 1)); te_max = np.nanmax(Y_te_full, axis=(0, 1))
            te_rng = np.maximum(te_max - te_min, 1e-8)

            te_preds_accum = []; tr_preds_accum = []; val_pos_accum = []
            for inner_fold in range(1, 6):
                in_tr = (tr_fold_assign >= 1) & (tr_fold_assign <= 5) & (tr_fold_assign != inner_fold)
                in_val = (tr_fold_assign == inner_fold)
                in_tr_idx = tr_idx[in_tr]; in_val_idx = tr_idx[in_val]
                Y_in_tr = Y_all[in_tr_idx][:, keep_idx, :]
                Ys_in_tr = (Y_in_tr - tr_min) / tr_rng
                Ys_in_val = (Y_all[in_val_idx][:, keep_idx, :] - tr_min) / tr_rng
                Ys_te = (Y_all[te_idx][:, keep_idx, :] - te_min) / te_rng

                ode = train_hyper(Ys_in_tr, Ys_in_val, S_all[in_tr_idx], S_all[in_val_idx], times_t, is_iter)
                ode_tr = ode_predict(ode, S_all[in_tr_idx], Ys_in_tr, times_t, is_iter)
                ode_val = ode_predict(ode, S_all[in_val_idx], Ys_in_val, times_t, is_iter)
                ode_te = ode_predict(ode, S_all[te_idx], Ys_te, times_t, is_iter)
                pred_val = ode_val * tr_rng + tr_min
                pred_te = ode_te * te_rng + te_min

                tr_preds_accum.append(pred_val)
                val_pos_accum.append(np.where(in_val)[0])
                te_preds_accum.append(pred_te)
                if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                print(f'    inner {inner_fold} done', end=' ', flush=True)
            print('', flush=True)

            # --- test (fold 6), ensemble of inner models ---
            pred_raw_te = np.mean(np.stack(te_preds_accum), axis=0)
            true_raw_te = Y_all[te_idx][:, keep_idx, :]
            pred_raw_te[:, 0, :] = true_raw_te[:, 0, :]          # 剔 t0: h0 强制真值
            # --- train OOF ---
            true_raw_tr = Y_all[tr_idx][:, keep_idx, :]
            pred_raw_tr = np.zeros_like(true_raw_tr)
            for rel_pos, pv in zip(val_pos_accum, tr_preds_accum):
                pred_raw_tr[rel_pos] = pv
            pred_raw_tr[:, 0, :] = true_raw_tr[:, 0, :]

            # train metrics (train-stats 归一化)
            true_norm_tr = (true_raw_tr - tr_min) / tr_rng
            pred_norm_tr = (pred_raw_tr - tr_min) / tr_rng
            pcc_tr, r2_tr, nr_tr, ms_tr, msn_tr, rl_tr, rows_tr = compute_metrics(
                pred_raw_tr, true_raw_tr, keep_times, trait_cols,
                true_norm=true_norm_tr, pred_norm=pred_norm_tr)
            for row in rows_tr: row['foldset'] = foldset + 1; row['mode'] = mode; row['split'] = 'train'
            all_trait_rows.extend(rows_tr)

            # test metrics (test-stats 归一化, dGP 口径)
            true_norm_te = (true_raw_te - te_min) / te_rng
            pred_norm_te = (pred_raw_te - te_min) / te_rng
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
            print(f'\n  === {mode} [{split}] RESULTS ===', flush=True)
            print(f'  pcc={mpcc:.4f}  R2m={mr2med:.2f}  MSE_norm={mmsn:.6f}  NRMSE={mnr:.4f}  r_long={mrl:.4f}  [{elapsed} min]', flush=True)
            all_summary.append({'mode': mode, 'split': split, 'pcc': mpcc, 'R2_median': mr2med,
                                'MSE_norm': mmsn, 'NRMSE': mnr, 'r_long': mrl, 'elapsed_min': elapsed})

        if all_trait_rows:
            pd.DataFrame(all_trait_rows).to_csv(OUT_DIR / f'knode_hyper_{mode}_arab_per_trait{_TAG}.csv', index=False)

    pd.DataFrame(all_detail).to_csv(OUT_DIR / f'knode_hyper_ode_arab_all_detail{_TAG}.csv', index=False)
    pd.DataFrame(all_summary).to_csv(OUT_DIR / f'knode_hyper_ode_arab_all_summary{_TAG}.csv', index=False)

    print('\n' + '=' * 56)
    print('ABLATION SUMMARY (hyper family, arab, train / test split)')
    print('=' * 56)
    print(f'{"Mode":>12s}  {"Split":>5s}  {"PCC":>8s}  {"R2m":>8s}  {"MSE_norm":>10s}  {"NRMSE":>7s}  {"Time":>6s}')
    for row in all_summary:
        print(f'{row["mode"]:>12s}  {row["split"]:>5s}  {row["pcc"]:8.4f}  {row["R2_median"]:8.2f}  '
              f'{row["MSE_norm"]:10.6f}  {row["NRMSE"]:7.4f}  {row["elapsed_min"]:5.1f}m')
    print('\nDone.')


if __name__ == '__main__':
    main()
