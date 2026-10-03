# -*- coding: utf-8 -*-
"""make_component_ablation_from_summary.py -- 用当前口径(训练折 min-max)的 _linear/_affine 运行
重建组件消融表 (protocol, variant, pcc, R2_median, MSE_norm, NRMSE, r_long)。

数据源 = 各运行自己写的 <prefix>_all_summary_{linear,affine}.csv 的 test 行(脚本内聚合口径,
剔首个时间点), 因此与主结果/基线/作者方法同空间。图脚本 fig_ablation_dual.py 只用 linear/affine
两档(variants = ['linear','affine'])。

用法: python make_component_ablation_from_summary.py --dataset maize|arab [--indir DIR] [--out CSV]
"""
import argparse, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = {'maize': 'knode_tp1', 'arab': 'knode_arab_tp1'}
DEFAULT_IN = {'maize': os.path.normpath(os.path.join(HERE, '..', '..', '..', 'outputs', 'reproduce', 'abl_shards')),
              'arab':  os.path.normpath(os.path.join(HERE, '..', '..', '..', 'outputs', 'reproduce', 'abl_shards'))}
DEFAULT_OUT = {'maize': os.path.join(HERE, 'component_ablation.csv'),
               'arab':  os.path.join(HERE, 'arab', 'figure9_arab_component_ablation.csv')}
# 允许直接指向 9/13 抓下来的快照名
# (drift, mode) -> 该分片跑完时抓下来的 summary 快照
SNAP = {'maize': {('linear', 'ode_hyper'): 'summary_linear_l_rec.csv',
                  ('linear', 'iter_hyper'): 'summary_linear_l_iter.csv',
                  ('affine', 'ode_hyper'): 'summary_affine_a_rec.csv',
                  ('affine', 'iter_hyper'): 'summary_affine_a_iter.csv'},
        'arab':  {('linear', 'ode_hyper'): 'summary_linear_arab_l_rec.csv',
                  ('linear', 'iter_hyper'): 'summary_linear_arab_l_iter.csv',
                  ('affine', 'ode_hyper'): 'summary_affine_arab_a_rec.csv',
                  ('affine', 'iter_hyper'): 'summary_affine_arab_a_iter.csv'}}
MODE2PROT = {'ode_hyper': 'recursive', 'iter_hyper': 'iterative'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['maize', 'arab'], required=True)
    ap.add_argument('--indir', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    indir = a.indir or DEFAULT_IN[a.dataset]
    out = a.out or DEFAULT_OUT[a.dataset]

    rows = []
    for drift in ['linear', 'affine']:
        for mode, prot in MODE2PROT.items():
            path = os.path.join(indir, SNAP[a.dataset][(drift, mode)])
            s = pd.read_csv(path)
            s = s[(s.split == 'test') & (s['mode'] == mode)]
            if s.empty:
                raise ValueError('no test row for %s/%s in %s' % (drift, mode, path))
            r = s.iloc[0]
            rows.append(dict(protocol=prot, variant=drift,
                             pcc=float(r['pcc']), R2_median=float(r['R2_median']),
                             MSE_norm=float(r['MSE_norm']), NRMSE=float(r['NRMSE']),
                             r_long=float(r['r_long'])))
    d = pd.DataFrame(rows)
    d['variant'] = pd.Categorical(d['variant'], categories=['linear', 'affine'], ordered=True)
    d['protocol'] = pd.Categorical(d['protocol'], categories=['recursive', 'iterative'], ordered=True)
    d = d.sort_values(['protocol', 'variant']).reset_index(drop=True)
    if a.dataset == 'arab':          # 与原 arab 消融表同列
        d = d[['protocol', 'variant', 'pcc', 'R2_median', 'MSE_norm', 'NRMSE']]
    d.to_csv(out, index=False)
    print(d.round(4).to_string(index=False))
    print('\nsaved ->', out)


if __name__ == '__main__':
    main()
