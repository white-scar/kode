"""
knode_arab_spectral_diagnostic.py — 拟南芥 K-mode 谱拟合诊断 (与玉米同协议)

问题 (用户 2026-08-17): 换数据集后第一步 = 谱诊断, 确认轨迹在指数谱上是否稀疏、K 饱和在哪。
  拟南芥: N=382, T=14 (DAS 8-21 每日等间隔), P=45。时间点数比玉米(25)少, K 上限受限。

方法 (与玉米 knode_spectral_diagnostic.py 完全一致): 共享 λ_k (全局) + 每(基因型,性状)OLS 权重。
  重建是"光谱能表达多少信息"的**上界** (W 用真轨迹 OLS, 非预测)。

输出: outputs/reproduce/knode_arab_spectral_diagnostic.json
用法: python knode_arab_spectral_diagnostic.py [--Ks 1,2,4,6,8,10]
"""
import argparse, json
import numpy as np, pandas as pd, torch
from scipy.stats import pearsonr
from pathlib import Path

DATA_DIR = Path('../data/arabidopsis')
OUT_DIR = Path('../outputs/reproduce')
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device: {DEVICE}', flush=True)

# ---- 数据加载 (同 knode_arabidopsis_np_ablation.py) ----
pheno = pd.read_csv(DATA_DIR / 'BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv')
cv_raw = pd.read_csv(DATA_DIR / 'cross_validation_folds_w_validation_set_arabidopsis.csv', index_col=0)
meta = ['bio_ID', 'accession', 'DAS']
trait_cols = [c for c in pheno.columns if c not in meta and not c.startswith('Unnamed') and c != '']
P = len(trait_cols)
times_all = np.array(sorted(pheno['DAS'].unique()), dtype=np.float64)
T = len(times_all)
common_ids = sorted(set(pheno['bio_ID'].astype(str)) & set(cv_raw.index))
N = len(common_ids)

Y_all = np.full((N, T, P), np.nan, dtype=np.float64)
for pi, trt in enumerate(trait_cols):
    sub = pheno[['bio_ID', 'DAS', trt]].copy(); sub['bio_ID'] = sub['bio_ID'].astype(str)
    Y_all[:, :, pi] = sub.pivot(index='bio_ID', columns='DAS', values=trt).loc[common_ids].values
assert not np.any(np.isnan(Y_all))
print(f'  N={N}  T={T}  P={P}  DAS={times_all.tolist()}', flush=True)


def fit_spectral(Y_n, Tn, K, iters=400, lr=2e-2):
    """共享 λ + 每(基因型,性状) OLS 权重; 返回重建轨迹 (N,T,P) 与学到的 λ."""
    torch.manual_seed(0)
    Y = torch.tensor(Y_n.astype(np.float32), device=DEVICE)            # (N,T,P)
    Yt = Y.permute(2, 1, 0)                                            # (P,T,N)
    lam = torch.nn.Parameter(torch.linspace(-8, 0.5, K, device=DEVICE))
    opt = torch.optim.Adam([lam], lr=lr)
    for _ in range(iters):
        Phi = torch.cat([torch.ones(Tn, 1, device=DEVICE),
                         torch.exp(lam * torch.linspace(0, 1, Tn, device=DEVICE)[:, None])], dim=1)  # (T,K+1)
        pinv = torch.linalg.pinv(Phi)                                  # (K+1,T)
        W = pinv @ Yt                                                  # (P,K+1,N)
        rec = (Phi @ W).permute(2, 1, 0)                               # (N,T,P)
        loss = ((rec - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Phi = torch.cat([torch.ones(Tn, 1, device=DEVICE),
                         torch.exp(lam.detach() * torch.linspace(0, 1, Tn, device=DEVICE)[:, None])], dim=1)
        W = torch.linalg.pinv(Phi) @ Yt
        rec = (Phi @ W).permute(2, 1, 0)
    return rec.cpu().numpy(), lam.detach().cpu().numpy()


def r_node(pred, Y_n):
    rs = np.full((pred.shape[1], pred.shape[2]), np.nan)
    for t in range(pred.shape[1]):
        for p in range(pred.shape[2]):
            if pred[:, t, p].std() > 1e-9 and Y_n[:, t, p].std() > 1e-9:
                rs[t, p] = pearsonr(pred[:, t, p], Y_n[:, t, p])[0]
    return float(np.nanmean(rs[1:]))        # 剔t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ks', default='1,2,4,6,8,10')
    a = ap.parse_args()
    Ks = [int(x) for x in a.Ks.split(',') if x.strip()]

    fc = cv_raw.loc[common_ids].iloc[:, 0].values.astype(int)
    tr = np.where((fc >= 1) & (fc <= 5))[0]; te = np.where(fc == 6)[0]
    tr_min = np.nanmin(Y_all[tr], axis=(0, 1)); tr_max = np.nanmax(Y_all[tr], axis=(0, 1))
    tr_rng = np.maximum(tr_max - tr_min, 1e-8)
    Y_tr_n = ((Y_all[tr] - tr_min) / tr_rng).astype(np.float32)
    Y_te_n = ((Y_all[te] - tr_min) / tr_rng).astype(np.float32)
    print(f'fold-set 0: train n={len(tr)}, test n={len(te)}  K={Ks}', flush=True)
    print(f'{"K":4s} | {"train重建r(剔t0)":>16s} {"test重建r(剔t0)":>15s} | 学到的λ(升序)', flush=True)

    out = {}
    for K in Ks:
        rec_tr, lam_tr = fit_spectral(Y_tr_n, T, K)
        rec_te, lam_te = fit_spectral(Y_te_n, T, K)
        r_tr = r_node(rec_tr, Y_tr_n); r_te = r_node(rec_te, Y_te_n)
        lam_s = ' '.join(f'{x:.2f}' for x in np.sort(lam_tr))
        out[K] = {'train_recon_r': r_tr, 'test_recon_r': r_te,
                  'lambda_train': lam_tr.tolist(), 'lambda_test': lam_te.tolist()}
        print(f'{K:4d} | {r_tr:16.4f} {r_te:15.4f} | {lam_s}', flush=True)

    print('\n[参照-玉米] K=1 -> 0.871, K=6 -> 0.947 饱和')
    print('[判断] 拟南芥 K 饱和在哪 -> 决定 spectral_K 的 K; 若连 K=8 都不饱和则谱方法不适用')
    with open(OUT_DIR / 'knode_arab_spectral_diagnostic.json', 'w', encoding='utf-8') as f:
        json.dump({'N': N, 'T': T, 'P': P, 'Ks': Ks, 'results': out}, f,
                  ensure_ascii=False, indent=2, default=float)
    print('Saved: knode_arab_spectral_diagnostic.json', flush=True)


if __name__ == '__main__':
    main()
