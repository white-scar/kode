# -*- coding: utf-8 -*-
"""拟南芥 KNode tp1skip 十折逐点整理: 4 模式 × 14 时点 (test), 先 trait 平均再跨 foldset mean±sd."""
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')

OUT = 'outputs/reproduce'
modes = ['ode_hyper', 'ode_hyper_skip', 'iter_hyper', 'iter_hyper_skip']
fmap = {
    'ode_hyper': 'knode_arab_tp1skip_ode_hyperarab_full10_dim30_per_trait.csv',
    'ode_hyper_skip': 'knode_arab_tp1skip_ode_hyper_skiparab_full10_dim30_per_trait.csv',
    'iter_hyper': 'knode_arab_tp1skip_iter_hyperarab_full10_dim30_per_trait.csv',
    'iter_hyper_skip': 'knode_arab_tp1skip_iter_hyper_skiparab_full10_dim30_per_trait.csv',
}
pt_all = {}
for m in modes:
    df = pd.read_csv(f'{OUT}/{fmap[m]}')
    df = df[df['split'] == 'test']
    pt_all[m] = df

# 逐时点: 先 trait 平均 (per mode+timepoint+foldset) -> 跨 foldset mean±sd
tp = sorted(pt_all['ode_hyper']['timepoint'].unique())
print(f'时点数: {len(tp)} ({tp[0]}–{tp[-1]}), 性状: {pt_all["ode_hyper"]["trait"].nunique()}, foldset: 10\n')

md_rows = []
for m in modes:
    fl = pt_all[m].groupby(['timepoint', 'foldset'])['pcc'].mean().reset_index()
    g = fl.groupby('timepoint')['pcc'].agg(['mean', 'std'])
    md_rows.append((m, g))

# 交叉验证 detail 文件
detail = pd.read_csv(f'{OUT}/knode_arab_tp1skip_all_detailarab_full10_dim30.csv')
det_te = detail[detail['split'] == 'test']
print('detail 时点:', sorted(det_te['timepoint'].unique()))
print('detail test pcc 是否 = per_trait 聚合 mean (ode_hyper_skip, 交集时点):')
g_ok = md_rows[1][1]['mean']
det_idx = det_te[det_te['mode'] == 'ode_hyper_skip'].set_index('timepoint')['pcc']
common = [t for t in tp if t in det_idx.index]
match = all(abs(g_ok[t] - det_idx.loc[t]) < 1e-9 for t in common)
print('  match:', match, ' (n=', len(common), ')')
print()

# 表格: 行=时点, 列=4模式 mean±sd
print('| timepoint | ode_hyper | ode_hyper_skip | iter_hyper | iter_hyper_skip |')
print('|---|---|---|---|---|')
for t in tp:
    cells = []
    for m, g in md_rows:
        mean = g.loc[t, 'mean']; sd = g.loc[t, 'std']
        if t == tp[0]:
            cells.append(f'**1.000** (锚定)')
        else:
            cells.append(f'{mean:.3f}±{sd:.3f}')
    print(f'| {t} | ' + ' | '.join(cells) + ' |')

# 全局 (排除 t0): per-time mean 的 foldset 语义 -> 直接用 all cells 平均
print()
print('全局 test (排除 t0, per-trait-cell 直接平均):')
for m, g in md_rows:
    te = pt_all[m]
    v = te[te['timepoint'] != tp[0]]['pcc']
    print(f'  {m:16s}: {v.mean():.4f} ± {v.std():.4f}')

# summary csv 对照
print()
print('summary csv (同文件):')
summ = pd.read_csv(f'{OUT}/knode_arab_tp1skip_all_summaryarab_full10_dim30.csv')
for _, r in summ[summ['split'] == 'test'].iterrows():
    print(f'  {r["mode"]:16s}: pcc={r["pcc"]:.4f}  R2_med={r["R2_median"]:.4f}  NRMSE={r["NRMSE"]:.4f}')
