"""knode_amplitude_heritability_arab.py — ① 幅度遗传可预测性诊断
用户 2026-08-18: 先确认"遗传信号是否集中在幅度" -> 决定能否再榨 PCC.
对照: 谱形状权重 W 的遗传可预测性 ~0.105 (memory). 若数据级幅度明显更高 -> 有肉.

幅度定义 (逐基因型标量):
  fin  第21天终值 (性状均值)
  inc  第21天 - 第8天 增量
  auc  曲线下面积 (trapezoid, 性状均值)
  Wg   谱生长模态权重 (6模态固定λ OLS, 性状均值)  <- 谱意义的幅度
  W_all  6个谱权重坐标的 GBLUP 平均 r           <- 形状权重可预测性对照 (~0.105)

方法: GBLUP = kernel ridge (核=GRM 382x382), KFold(5, shuffle, seed42) CV,
      报逐目标 pooled Pearson r. 所有目标用同一 λ 规则, 公平对比.
输出: outputs/reproduce/knode_amplitude_heritability_arab.json
"""
import json
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from pathlib import Path

import knode_arab_data as sg

OUT_DIR = sg.OUT_DIR

# ---- 全 GRM 作为核 ----
grm_raw = pd.read_csv('../data/arabidopsis/arabidopsis_GRM_382.csv', index_col=0)
grm_raw.index = grm_raw.index.astype(str); grm_raw.columns = grm_raw.columns.astype(str)
Kmat = grm_raw.loc[sg.common_ids, sg.common_ids].values.astype(np.float64)   # (N,N)
N, T, P = sg.Y_all.shape
print(f'N={N} T={T} P={P}', flush=True)

Y = sg.Y_all
# 每性状跨基因型/时间标准化 (供谱权重拟合用)
Ys = (Y - Y.mean(axis=(0, 1))) / (Y.std(axis=(0, 1)) + 1e-8)

# ---- 幅度目标 (在标准化表型 Ys 上, 与谱权重同尺度, 公平对比) ----
fin = Ys[:, -1, :].mean(axis=1)                       # (N,) 终值
inc = (Ys[:, -1, :] - Ys[:, 0, :]).mean(axis=1)       # 增量
auc = np.trapezoid(Ys, axis=1).mean(axis=1)           # 曲线下面积
# 相对共享形状的幅度系数: Ys_g ≈ a_g * 均值形状 (λ共享/幅度故事里的幅度)
vec_mean = Ys.mean(axis=0)                            # (T,P) 共享形状
scale = np.sum(Ys * vec_mean[None, :], axis=(1, 2)) / (np.sum(vec_mean ** 2) + 1e-8)

# 谱权重 (固定 λ = 数据层诊断 K=6 学到的值), 逐(基因型,性状) OLS, 再性状均值
lam = np.array([-7.78, -6.53, -4.36, -3.11, -1.02, 0.69])
tn = np.linspace(0, 1, T)
Phi = np.hstack([np.ones((T, 1)), tn[:, None], np.exp(np.outer(tn, lam))])  # (T,8)
pinv = np.linalg.pinv(Phi)                              # (8,T)
W_gt = np.einsum('kt,ntp->nkp', pinv, Ys)              # (N,8,P) 逐(基因型,性状)谱权重
W_all = W_gt.mean(axis=2)                              # (N,8) 性状均值
Wg = W_all[:, -1]                                      # 生长模态权重

targets = {'fin_终值(标准)': fin, 'inc_增量(标准)': inc, 'auc_面积(标准)': auc,
           'scale_相对形状幅度': scale,
           'Wg_谱生长模态': Wg, 'W1..8_谱形状权重': W_all}


def gblup_cv(z, Kmat, folds=5, seed=42):
    """kernel ridge (GBLUP) CV, 返回 pooled Pearson r."""
    kf = KFold(folds, shuffle=True, random_state=seed)
    lam_reg = 0.1 * np.trace(Kmat) / len(z)             # 固定收缩规则, 各目标公平
    yp = np.zeros_like(z)
    for tr, te in kf.split(z):
        ktr = Kmat[np.ix_(tr, tr)]; kte = Kmat[np.ix_(te, tr)]
        alpha = np.linalg.solve(ktr + lam_reg * np.eye(len(tr)), z[tr])
        yp[te] = kte @ alpha
    r = float(np.corrcoef(z, yp)[0, 1])
    return r, yp


print(f'{"target":18s} | GBLUP CV r | 说明', flush=True)
out = {'N': N, 'method': 'GBLUP kernel-ridge, KFold5 seed42, lam_reg=0.1*trace/N', 'results': {}}
for name, z in targets.items():
    if z.ndim == 1:
        r, yp = gblup_cv(z, Kmat)
        out['results'][name] = {'r': round(r, 4)}
        print(f'{name:18s} | {r:8.4f}', flush=True)
    else:                                               # (N,8) 逐坐标预测, 平均 r
        rs = []
        for j in range(z.shape[1]):
            rj, _ = gblup_cv(z[:, j], Kmat)
            rs.append(rj)
        mean_r = float(np.mean(rs))
        out['results'][name] = {'r_each': [round(x, 4) for x in rs], 'r_mean': round(mean_r, 4)}
        print(f'{name:18s} | {mean_r:8.4f}  (逐坐标 {[round(x,3) for x in rs]})', flush=True)

print('\n[对照-memory] 谱形状权重 W 遗传可预测性 ~0.105')
with open(OUT_DIR / 'knode_amplitude_heritability_arab.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=float)
print('Saved: knode_amplitude_heritability_arab.json', flush=True)
