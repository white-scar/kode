#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-panel Kode component ablation: PCC and log10(MSE), Arabidopsis."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
CSV = 'figure9_arab_component_ablation.csv'
OUT = 'Figure9_arab_component_ablation.svg'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10.5,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'savefig.dpi': 300,
    'figure.dpi': 110,
    'svg.fonttype': 'none',
})

STYLE = {
    'recursive': dict(color='#D62728', ls='-', marker='o', label='Recursive'),
    'iterative': dict(color='#2B2B2B', ls=(0, (5.0, 2.2)), marker='s', label='Iterative'),
}


def bold_ticks(ax, size=9):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight('bold')
        lab.set_fontsize(size)


def main():
    df = pd.read_csv(CSV)
    if 'mse' not in df.columns and 'MSE_norm' in df.columns:
        df['mse'] = df['MSE_norm']
    variants = [v for v in ['linear', 'affine'] if v in set(df['variant'])]
    protocols = [p for p in ['recursive', 'iterative'] if p in set(df['protocol'])]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    panels = [('pcc', 'Prediction accuracy (PCC)', False),
              ('mse', r'$\log_{10}$(MSE)', True)]

    for ax, (metric, ylabel, use_log) in zip(axes, panels):
        x = np.arange(len(variants), dtype=float)
        for protocol in protocols:
            st = STYLE[protocol]
            sub = df[df['protocol'] == protocol].set_index('variant')
            y = []
            for v in variants:
                val = float(sub.loc[v, metric]) if v in sub.index else np.nan
                y.append(np.log10(val + 1e-12) if use_log and np.isfinite(val) else val)
            ax.plot(x, y, color=st['color'], ls=st['ls'], lw=2.0,
                    marker=st['marker'], ms=4.8, label=st['label'], zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=18, ha='right', fontsize=10.5, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', length=3, labelsize=9)
        bold_ticks(ax, 9)
        if use_log:
            ax.set_ylim(-3.0, 0.3)
            ax.set_yticks([-3, -2, -1, 0])

    axes[0].text(-0.18, 1.04, 'a', transform=axes[0].transAxes,
                 fontsize=20, fontweight='bold')
    axes[1].text(-0.18, 1.04, 'b', transform=axes[1].transAxes,
                 fontsize=20, fontweight='bold')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.55, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT, bbox_inches='tight')
    print('saved', OUT)


if __name__ == '__main__':
    main()