# -*- coding: utf-8 -*-
"""逐时点对比: 动态基线族(6 fams) vs 我的 ode_hyper_skip(rec), 均 maize_magic test split.
PCC 对每性状仿射变换不变 -> 预测(归一化空间) 与 原始观测 直接 pearsonr 即可.
"""
import json, numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from pathlib import Path

DATA_DIR = Path('data/maize_magic')
OUT_DIR = Path('outputs/reproduce')

# ---- 重建 Y_all (与 knode_maize_np_tp1_affine.load_data 一致) ----
pheno = pd.read_csv(DATA_DIR / 'maize_trait_BLUP_50_NArmv_traits_afterClustering.csv')
cv_raw = pd.read_csv(DATA_DIR / 'cross_validation_folds_w_validation_set_maize.csv', index_col=0)
meta_cols = ['bio_ID', 'accession', 'DAS']
trait_cols = [c for c in pheno.columns if c not in meta_cols and not c.startswith('Unnamed') and c != '']
P = len(trait_cols)
times_all = np.array(sorted(pheno['DAS'].unique()), dtype=np.float64)
T = len(times_all)
common_ids = sorted(set(pheno['bio_ID'].astype(str)) & set(cv_raw.index))
N = len(common_ids)
Y_all = np.full((N, T, P), np.nan)
for pi, trait in enumerate(trait_cols):
    sub = pheno[['bio_ID', 'DAS', trait]].copy()
    sub['bio_ID'] = sub['bio_ID'].astype(str)
    mat = sub.pivot(index='bio_ID', columns='DAS', values=trait).loc[common_ids]
    Y_all[:, :, pi] = mat.values
assert not np.any(np.isnan(Y_all))
print(f'Y_all: {N} genotypes x {T} times x {P} traits;  trait_cols[0]={trait_cols[0]}')

das_map = {float(t): int(t) for t in times_all}   # DAS 都是 int 值 (15..47)

# ---- json: test_ids + fams + iters ----
j = json.load(open(OUT_DIR / 'knode_dynamic_baselines_full.json'))
fams = j['fams']; iters = j['iters']; test_ids_by_iter = j['test_ids_by_iter']
print('fams:', fams, ' iters:', iters)

id2idx = {cid: i for i, cid in enumerate(common_ids)}

# ---- 逐 (t,p) PCC 的 per-time mean ----
def per_time_pcc(pred, Y_te):
    """pred, Y_te: (n_te, T, P) -> r[T,P] -> per-time mean over P."""
    r = np.full((T, P), np.nan)
    for t in range(T):
        for p in range(P):
            r[t, p] = pearsonr(pred[:, t, p], Y_te[:, t, p])[0]
    return r.mean(axis=1)  # (T,)

d = np.load(OUT_DIR / 'knode_dynamic_baselines_full_preds.npz', allow_pickle=True)

per_iter_time = {fam: np.zeros((len(iters), T)) for fam in fams}
for k in iters:
    te_ids = test_ids_by_iter[str(k)]
    te_idx = np.array([id2idx[i] for i in te_ids])
    Y_te = Y_all[te_idx]  # (55, T, P) 原始观测
    for fam in fams:
        pred = d[f'{fam}_iter{k}_avg']  # (55, T, P)
        per_iter_time[fam][k] = per_time_pcc(pred, Y_te)

# ---- 我的 ode_hyper_skip (rec) 逐时点 (跨 foldset 的 per_trait) ----
pt = pd.read_csv(OUT_DIR / 'knode_tp1skip_ode_hyper_skip_per_trait.csv')
pt = pt[pt['split'] == 'test']
fl = pt.groupby(['mode', 'timepoint', 'foldset'])['pcc'].mean().reset_index()
my = fl[fl['mode'] == 'ode_hyper_skip'].groupby('timepoint')['pcc'].agg(['mean', 'std'])

# ---- 汇总: 每 fam 逐时点 mean±std(跨 iters) ----
rows = []
tp_order = sorted(times_all)
for fam in fams:
    m = per_iter_time[fam].mean(axis=0)
    s = per_iter_time[fam].std(axis=0)
    for i, t in enumerate(tp_order):
        rows.append(dict(model=fam, timepoint=int(t), DAS=das_map[t], mean=m[i], sd=s[i]))
# 我的方法 (foldset 维度 = 10, 与 iters 同协议)
for t, r in my.iterrows():
    rows.append(dict(model='ode_hyper_skip(rec,my)', timepoint=int(t), DAS=das_map[float(t)],
                     mean=r['mean'], sd=r['std']))

out = pd.DataFrame(rows)
out['mean'] = out['mean'].round(4); out['sd'] = out['sd'].round(4)
out.to_csv(OUT_DIR / '_compare_per_time.csv', index=False)

# 打印: 行=timepoint, 列=方法
wide = out.pivot_table(index='timepoint', columns='model', values='mean')
wide.insert(0, 'DAS', [das_map[float(t)] for t in wide.index])
pd.set_option('display.width', 250)
print('\n===== 逐时点 test 平均 PCC (mean over traits) =====')
print(wide.round(3).to_string())

# 全局 mean±sd (剔 t0) 汇总对比
print('\n===== 全局 (剔 t0) mean±sd =====')
summ = []
for fam in fams + ['ode_hyper_skip(rec,my)']:
    mm = out[(out['model'] == fam) & (out['timepoint'] != 15)]['mean']
    summ.append((fam, mm.mean(), mm.std()))
summ = pd.DataFrame(summ, columns=['model', 'mean', 'sd']).round(4)
print(summ.to_string(index=False))
print('\nSaved: outputs/reproduce/_compare_per_time.csv')
