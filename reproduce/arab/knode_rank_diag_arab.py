"""knode_rank_diag_arab.py — 算子低秩诊断 (拟南芥, 探针).
一次扫多个秩 K: 对每秩训练标准 Koopman ODE (线性 cond->(u,v)), 报告:
  (1) 性能饱和曲线 r(K): 若 K=2 就平台期, 说明更多秩对预测无益 (低秩足够的模型选择论证);
  (2) 每基因型算子 M_i = U_i V_i^T 的 σ 谱 (非零奇异值 = singular values(V^T U, KxK)):
      σ_3/σ_1 是否趋零, 判断"无约束算子在数据驱动下是否自发低秩" (注意高秩过拟合混淆);
  (3) 主导模态特征值 λ 的跨基因型分布 (时间常数是否共享).
用法: python knode_rank_diag_arab.py --ranks 1,2,3,4,6,8,15 --seeds 42 --iters 0 --grm_dim 40
输出: outputs/reproduce/knode_rank_diag_arab.json
"""
import argparse, json
import numpy as np, torch
from knode_cond_variants_arab import train_variant, predict_variant, _r_nrmse
import knode_arab_data as sg

OUT_DIR = sg.OUT_DIR
DEVICE = sg.DEVICE


def analyze_operator(model, grm, tp1):
    """每基因型: SVD(V^T U) 的 σ 谱 + 特征值 λ. 返回 (B,K) 奇异值, (B,K) 复数特征值."""
    with torch.no_grad():
        g = torch.tensor(np.asarray(grm, dtype=np.float32), device=DEVICE)
        y = torch.tensor(np.asarray(tp1, dtype=np.float32), device=DEVICE)
        model.set_z(g, y)
        U = model._u; V = model._v           # (B,K,P)
    VtU = torch.einsum('brp,bkp->brk', V, U)  # (B,K,K): V^T U, 非零奇异值 = M 的非零奇异值
    s = torch.linalg.svdvals(VtU)
    lam = torch.linalg.eigvals(VtU)
    return s.detach().cpu().numpy(), lam.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ranks', default='1,2,3,4,6,8,15')
    ap.add_argument('--seeds', default='42')
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--iters', default='0')
    ap.add_argument('--grm_dim', type=int, default=40)
    a = ap.parse_args()
    ranks = [int(x) for x in a.ranks.split(',') if x.strip()]
    seeds = [int(x) for x in a.seeds.split(',') if x.strip()]
    iters = [int(x) for x in a.iters.split(',') if x.strip()]
    sg.GRM_DIM = a.grm_dim
    print(f'Device:{DEVICE} ranks={ranks} seeds={seeds} iters={iters} grm_dim={a.grm_dim}', flush=True)

    out = {'ranks': ranks, 'seeds': seeds, 'iters': iters, 'grm_dim': a.grm_dim, 'per_rank': {}}
    for K in ranks:
        for k in iters:
            fc = sg.cv_raw.loc[sg.common_ids].iloc[:, k].values.astype(int)
            tr = np.where((fc >= 1) & (fc <= 5))[0]; te = np.where(fc == 6)[0]
            tr_min = np.nanmin(sg.Y_all[tr], axis=(0, 1)); tr_max = np.nanmax(sg.Y_all[tr], axis=(0, 1))
            tr_rng = np.maximum(tr_max - tr_min, 1e-8)
            te_min = np.nanmin(sg.Y_all[te], axis=(0, 1)); te_max = np.nanmax(sg.Y_all[te], axis=(0, 1))
            te_rng = np.maximum(te_max - te_min, 1e-8)
            Ys_tr = ((sg.Y_all[tr] - tr_min) / tr_rng).astype(np.float32)
            y1_all = ((sg.Y_all[:, 0, :] - tr_min) / tr_rng).astype(np.float32)
            Y_te_n = ((sg.Y_all[te] - te_min) / te_rng).astype(np.float32)
            S_tr = sg.get_S(a.grm_dim)[tr]; S_te = sg.get_S(a.grm_dim)[te]
            obs = np.arange(1, sg.T)

            m = train_variant('baseline', Ys_tr, S_tr, y1_all[tr], seeds[0], a.epochs, K, a.grm_dim)
            p = predict_variant(m, S_te, y1_all[te])
            p = ((p * tr_rng + tr_min - te_min) / te_rng).astype(np.float32)
            r_all, _ = _r_nrmse(p, Y_te_n, np.arange(sg.T))
            r_tt, nm_tt = _r_nrmse(p, Y_te_n, obs)

            s_te, lam_te = analyze_operator(m, S_te, y1_all[te])

            sm = s_te.mean(axis=0); sv = s_te.std(axis=0)
            order = np.argsort(-np.abs(lam_te), axis=1)
            lam_sorted = np.take_along_axis(lam_te, order, axis=1)
            re = lam_sorted.real

            print(f'\n===== rank={K} iteration {k} =====', flush=True)
            print(f'  剔t0 r={r_tt:.4f} 含t0 r={r_all:.4f} NRMSE={nm_tt:.4f}', flush=True)
            print('  σ 谱: ' + '  '.join(f'σ{i+1}={sm[i]:.3f}' for i in range(min(K, 8))), flush=True)
            if K >= 3:
                print(f'  比值: σ2/σ1={sm[1]/max(sm[0],1e-9):.3f}  '
                      f'σ3/σ1={sm[2]/max(sm[0],1e-9):.3f}  '
                      f'σ4/σ1={sm[3]/max(sm[0],1e-9):.3f}' if K >= 4 else
                      f'  比值: σ2/σ1={sm[1]/max(sm[0],1e-9):.3f}  σ3/σ1={sm[2]/max(sm[0],1e-9):.3f}',
                      flush=True)
            print('  Re(λ) 前几个 (mean±std over genotypes): ' +
                  '  '.join(f'λ{i+1}={re[:, i].mean():+.3f}±{re[:, i].std():.3f}'
                            for i in range(min(K, 3))), flush=True)

            entry = {
                'metrics': {'r_含t0': r_all, 'r_剔t0': r_tt, 'nrmse_剔t0': nm_tt},
                'sigma_mean': sm[:8].tolist(), 'sigma_std': sv[:8].tolist(),
                'ratio': {f's{j}_s1': float(sm[j - 1] / max(sm[0], 1e-9))
                          for j in range(2, min(K + 1, 6))},
                're_lambda_mean': re[:, :min(K, 3)].mean(axis=0).tolist(),
                're_lambda_std': re[:, :min(K, 3)].std(axis=0).tolist(),
            }
            out['per_rank'][str(K)] = entry
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    with open(OUT_DIR / 'knode_rank_diag_arab.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f'\nSaved: knode_rank_diag_arab.json', flush=True)


if __name__ == '__main__':
    main()
