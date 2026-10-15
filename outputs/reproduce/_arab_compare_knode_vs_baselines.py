# -*- coding: utf-8 -*-
"""拟南芥: KNode tp1skip (4模式) vs 动态基线族 (6 fams) 逐时点同口径对比.
PCC 对仿射不变 -> npz 预测(归一化空间) 与 原始观测直接 pearsonr.
"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

OUT = 'outputs/reproduce'
DATA = Path('data/arabidopsis')

# ---- 重建 arab Y_all / common_ids (镜像 knode_arab_data) ----
pheno = pd.read_csv(DATA / 'BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv')
cv_raw = pd.read_csv(DATA / 'cross_validation_folds_w_validation_set_arabidopsis.csv', index_col=0)
meta_cols = ['bio_ID', 'accession', 'DAS']
trait_cols = [c for c in pheno.columns if c not in meta_cols and not c.startswith('Unnamed') and c != '']
P = len(trait_cols)
times_all = np.array(sorted(pheno['DAS'].unique()), dtype=np.float64)
T = len(times_all)
common_ids = sorted(set(pheno['bio_ID'].astype(str)) & set(cv_raw.index))
N = len(common_ids)
Y_all = np.full((N, T, P), np.nan, dtype=np.float64)
for pi, trait in enumerate(trait_cols):
    sub = pheno[['bio_ID', 'DAS', trait]].copy(); sub['bio_ID'] = sub['bio_ID'].astype(str)
    Y_all[:, :, pi] = sub.pivot(index='bio_ID', columns='DAS', values=trait).loc[common_ids].values
assert not np.any(np.isnan(Y_all))
print(f'arab: N={N} genotypes, T={T} times({times_all[0]}..{times_all[-1]}), P={P} traits')

j = json.load(open(f'{OUT}/knode_dynamic_baselines_full_arab.json'))
fams = j['fams']; iters = j['iters']; test_ids_by_iter = j['test_ids_by_iter']
d = np.load(f'{OUT}/knode_dynamic_baselines_full_arab_preds.npz', allow_pickle=True)
id2idx = {cid: i for i, cid in enumerate(common_ids)}

def per_time_pcc(pred, Y_te):
    r = np.full((T, P), np.nan)
    for t in range(T):
        for p in range(P):
            r[t, p] = pearsonr(pred[:, t, p], Y_te[:, t, p])[0]
    return r.mean(axis=1)

per_iter_time = {fam: np.zeros((len(iters), T)) for fam in fams}
for k in iters:
    te_idx = np.array([id2idx[i] for i in test_ids_by_iter[str(k)]])
    Y_te = Y_all[te_idx]
    for fam in fams:
        pred = d[f'{fam}_iter{k}_avg']
        per_iter_time[fam][k] = per_time_pcc(pred, Y_te)

# ---- KNode tp1skip 4 模式 (per_trait, 先 trait 平均再跨 foldset) ----
modes = ['ode_hyper', 'ode_hyper_skip', 'iter_hyper', 'iter_hyper_skip']
fmap = {
    'ode_hyper': 'knode_arab_tp1skip_ode_hyperarab_full10_dim30_per_trait.csv',
    'ode_hyper_skip': 'knode_arab_tp1skip_ode_hyper_skiparab_full10_dim30_per_trait.csv',
    'iter_hyper': 'knode_arab_tp1skip_iter_hyperarab_full10_dim30_per_trait.csv',
    'iter_hyper_skip': 'knode_arab_tp1skip_iter_hyper_skiparab_full10_dim30_per_trait.csv',
}
mine = {}
for m in modes:
    df = pd.read_csv(f'{OUT}/{fmap[m]}')
    df = df[df['split'] == 'test']
    fl = df.groupby(['timepoint', 'foldset'])['pcc'].mean().reset_index()
    g = fl.groupby('timepoint')['pcc'].agg(['mean', 'std'])
    mine[m] = g

# ---- 汇总表 ----
tp_order = sorted(times_all)
rows = []
for fam in fams:
    m = per_iter_time[fam].mean(axis=0); s = per_iter_time[fam].std(axis=0)
    for i, t in enumerate(tp_order):
        rows.append(dict(model=fam, timepoint=int(t), mean=m[i], sd=s[i], kind='baseline'))
for m, g in mine.items():
    for t, r in g.iterrows():
        rows.append(dict(model=m, timepoint=int(t), mean=r['mean'], sd=r['std'], kind='knode'))
out = pd.DataFrame(rows)
out.to_csv(f'{OUT}/_arab_compare_knode_vs_baselines_per_time.csv', index=False)

# 并排宽表
wide = out.pivot_table(index='timepoint', columns='model', values='mean').round(3)
print('\n===== 逐时点 test 平均 PCC (mean over traits) =====')
print(wide.to_string())
print('\n===== 全局 (剔 t0) mean±sd =====')
cols = ['ode_hyper_skip', 'ode_hyper', 'iter_hyper', 'iter_hyper_skip'] + fams
for m in cols:
    mm = out[(out['model'] == m) & (out['timepoint'] != int(tp_order[0]))]['mean']
    print(f'  {m:16s}: {mm.mean():.4f} ± {mm.std():.4f}')
