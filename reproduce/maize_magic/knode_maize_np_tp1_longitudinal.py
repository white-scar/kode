#!/usr/bin/env python3
"""KNode (affine) 玉米 MAGIC — longitudinal 精度 (论文口径, Hobby et al. 2026).

复用 knode_maize_np_tp1_affine.py 的协议/模型, 新增 longitudinal PCC:
  论文定义 (Methods, "Assessment of accuracy"): snapshot = 每 (trait×时点) 跨基因型 PCC;
  longitudinal = 每 (基因型, 性状) 一条时间轨迹的 PCC, 再跨全部 traits×genotypes×fold 聚合.
实现: 对每个测试基因型的每个性状, 用剔 t0 的 25→24 时点真实/预测轨迹算 pearsonr.
对照输出: snapshot (per-trait×timepoint 聚合, 原口径) 与 longitudinal.

用法: python knode_maize_np_tp1_longitudinal.py [--foldsets 1] [--modes ode_hyper,...]
"""
import argparse, sys, os, time
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import knode_maize_np_tp1_affine as k


def snapshot_pcc(pred_raw, true_raw):
    """原口径: 每 (时点, 性状) 跨基因型 PCC, 剔 t0, 全部平均."""
    n_g, T, n_p = pred_raw.shape
    rs = []
    for t in range(1, T):
        for pi in range(n_p):
            yp = pred_raw[:, t, pi]; yt = true_raw[:, t, pi]
            m = ~np.isnan(yp) & ~np.isnan(yt)
            if m.sum() >= 3:
                r = pearsonr(yt[m], yp[m])[0]
                if not np.isnan(r): rs.append(r)
    return float(np.mean(rs)) if rs else float('nan')


def longitudinal_pcc(pred_raw, true_raw, t_start=1):
    """论文 longitudinal: 每 (基因型, 性状) 轨迹 PCC (剔 t0), 返回全部 r 列表."""
    n_g, T, n_p = pred_raw.shape
    rs = []
    for gi in range(n_g):
        for pi in range(n_p):
            yp = pred_raw[gi, t_start:, pi]; yt = true_raw[gi, t_start:, pi]
            m = ~np.isnan(yp) & ~np.isnan(yt)
            if m.sum() >= 3:
                r = pearsonr(yt[m], yp[m])[0]
                if not np.isnan(r): rs.append(r)
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--foldsets', type=int, default=1)
    ap.add_argument('--modes', default='ode_hyper,iter_hyper')
    ap.add_argument('--epochs', type=int, default=k.N_EPOCHS)
    ap.add_argument('--drift', type=str, default='affine', choices=['linear', 'affine'])
    ap.add_argument('--tag', default='')
    a, _ = ap.parse_known_args()
    MODES = [m.strip() for m in a.modes.split(',')]
    k.N_EPOCHS = a.epochs
    DRIFT = a.drift

    D = k.load_data(a.foldsets)
    grm_pca, Y_all = D['grm_pca'], D['Y_all']
    cv = D['cv_foldsets']
    times_t = k.torch.tensor(D['tn'], dtype=k.torch.float32, device=k.DEVICE)
    T = D['T']; P = D['P']

    all_rows = []
    for mode in MODES:
        is_iter = 'iter' in mode
        t0 = time.time()
        for foldset in range(a.foldsets):
            fc = cv[:, foldset]
            tr_mask = (fc >= 1) & (fc <= 5); te_mask = (fc == 6)
            tr_idx = np.where(tr_mask)[0]; te_idx = np.where(te_mask)[0]
            tr_fold = fc[tr_idx]

            Y_tr = Y_all[tr_idx]; Y_te = Y_all[te_idx]
            tr_min = np.nanmin(Y_tr, axis=(0, 1)); tr_max = np.nanmax(Y_tr, axis=(0, 1))
            tr_rng = np.maximum(tr_max - tr_min, 1e-8)
            te_min = np.nanmin(Y_te, axis=(0, 1)); te_max = np.nanmax(Y_te, axis=(0, 1))
            te_rng = np.maximum(te_max - te_min, 1e-8)

            te_preds = []
            for inner in range(1, 6):
                in_tr = (tr_fold >= 1) & (tr_fold <= 5) & (tr_fold != inner)
                in_val = (tr_fold == inner)
                i_tr = tr_idx[in_tr]; i_val = tr_idx[in_val]
                Ys_tr = (Y_all[i_tr] - tr_min) / tr_rng
                Ys_val = (Y_all[i_val] - tr_min) / tr_rng
                Ys_te = (Y_te - te_min) / te_rng
                ode = k.train_hyper(Ys_tr, Ys_val, grm_pca[i_tr], grm_pca[i_val],
                                    times_t, is_iter, k.N_EPOCHS, DRIFT)
                p = k.ode_predict(ode, grm_pca[te_idx], Ys_te, times_t, is_iter)
                te_preds.append(p * te_rng + te_min)
                if k.DEVICE.type == 'cuda': k.torch.cuda.empty_cache()
            pred_raw = np.mean(np.stack(te_preds), axis=0)
            true_raw = Y_te.copy()
            pred_raw[:, 0, :] = true_raw[:, 0, :]   # t0 钉死 (同原脚本)

            sn = snapshot_pcc(pred_raw, true_raw)
            lg = longitudinal_pcc(pred_raw, true_raw, t_start=1)
            lg_all = longitudinal_pcc(pred_raw, true_raw, t_start=0)
            r_long_per_geno = [
                pearsonr(true_raw[gi].ravel()[~np.isnan(true_raw[gi].ravel())],
                         pred_raw[gi].ravel()[~np.isnan(true_raw[gi].ravel())])[0]
                for gi in range(true_raw.shape[0])]
            r_long_mean = float(np.mean(r_long_per_geno))
            all_rows.append({'mode': mode, 'foldset': foldset + 1,
                             'snapshot_pcc': sn,
                             'longitudinal_pcc_mean': float(np.mean(lg)),
                             'longitudinal_pcc_sd': float(np.std(lg)),
                             'longitudinal_pcc_median': float(np.median(lg)),
                             'n_traj': len(lg),
                             'longitudinal_incl_t0': float(np.mean(lg_all)),
                             'r_long_geno_flat': r_long_mean})
            print(f'  [{mode}] fs{foldset+1}: snap={sn:.4f}  longi(剔t0)={np.mean(lg):.4f}±{np.std(lg):.4f}  '
                  f'longi含t0={np.mean(lg_all):.4f}  r_long(展平)={r_long_mean:.4f}  n={len(lg)}', flush=True)
        print(f'  {mode}: {time.time()-t0:.0f}s', flush=True)

    df = pd.DataFrame(all_rows)
    out = Path(k.OUT_DIR) / f'knode_tp1_longitudinal{("_" + a.tag) if a.tag else ""}.csv'
    df.to_csv(out, index=False)
    print(f'\nSaved -> {out}')

    print('\n===== 汇总 (跨 foldset 均值) =====')
    g = df.groupby('mode')[['snapshot_pcc', 'longitudinal_pcc_mean', 'longitudinal_pcc_median',
                            'longitudinal_incl_t0', 'r_long_geno_flat']].mean().round(4)
    print(g.to_string())
    g.to_csv(Path(k.OUT_DIR) / f'knode_tp1_longitudinal_summary{("_" + a.tag) if a.tag else ""}.csv')


if __name__ == '__main__':
    main()
