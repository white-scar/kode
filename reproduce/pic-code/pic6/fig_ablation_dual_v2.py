#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One 2x3 ablation figure: each panel has PCC bars and MSE lines."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
COMP = HERE / 'component_ablation.csv'
HP = HERE / 'hyperparam_ablation.csv'
OUT = HERE / 'Figure_ablation_dual_v2.svg'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11.0,
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

A_COL = {'recursive': '#1F5FA8', 'iterative': '#9ECAE1'}
SINGLE_PCC = '#2C7FB8'
SINGLE_MSE = '#D62728'
MSE_LO, MSE_HI = -2.95, -1.80        # log10(MSE) range shared by all panels
MSE_TICKS = [-2.5, -2.0]


def bold_ticks(ax, size=9):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight('bold')
        lab.set_fontsize(size)


def panel_letter(ax, letter):
    ax.text(-0.18, 1.05, letter, transform=ax.transAxes,
            fontsize=22, fontweight='bold', va='bottom')


def style_base(ax, title, xlabel):
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', length=3, labelsize=9)
    bold_ticks(ax, 9)


def twin_axis(ax):
    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(axis='y', length=3, labelsize=9, colors=SINGLE_MSE)
    for lab in ax2.get_yticklabels():
        lab.set_fontweight('bold')
        lab.set_fontsize(9)
    return ax2


def annotate_bars(ax, xs, vals, color='black', fs=8.8):
    for x, v in zip(xs, vals):
        if np.isfinite(v):
            ax.annotate(f'{v:.3f}', (x, v), textcoords='offset points',
                        xytext=(0, 1.5), ha='center', va='bottom',
                        rotation=0, fontsize=fs, fontweight='bold', color=color)


def annotate_line(ax, xs, vals, color, fs=8.0, dy=3, fmt='{:.4f}'):
    for x, v in zip(xs, vals):
        if np.isfinite(v):
            ax.annotate(fmt.format(v), (x, v), textcoords='offset points',
                        xytext=(0, dy), ha='center', va='bottom',
                        fontsize=fs, fontweight='bold', color=color)


def component_panel(ax, comp):
    variants = ['linear', 'affine']
    labels = ['No affine', 'Affine']
    protocols = ['recursive', 'iterative']
    x = np.arange(2, dtype=float)
    w = 0.34
    for i, protocol in enumerate(protocols):
        d = comp[comp.protocol == protocol].set_index('variant')
        vals = [float(d.loc[v, 'pcc']) for v in variants]
        xs = x + (i - 0.5) * w
        ax.bar(xs, vals, width=w, color=A_COL[protocol], alpha=0.95,
               edgecolor='black', linewidth=0.3)
        annotate_bars(ax, xs, vals, color='black')
        for xi, vi in zip(xs, vals):
            frac = 0.45 if protocol == 'iterative' else 0.78
            ax.text(xi, vi * frac, protocol, rotation=0, ha='center', va='center',
                    fontsize=10.5, fontweight='bold', color='black')

    ax2 = twin_axis(ax)
    for protocol in protocols:
        d = comp[comp.protocol == protocol].set_index('variant')
        vals = [np.log10(float(d.loc[v, 'MSE_norm'])) for v in variants]
        ls = '-' if protocol == 'recursive' else (0, (5, 2))
        mk = 'o' if protocol == 'recursive' else 's'
        ax2.plot(x, vals, color=SINGLE_MSE, ls=ls, marker=mk,
                 ms=4.0, lw=1.6)
        annotate_line(ax2, x, vals, color=SINGLE_MSE, dy=3, fmt='{:.2f}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax2.set_ylim(MSE_LO, MSE_HI); ax2.set_yticks(MSE_TICKS)   # 2026-09-21: 适配当前口径 MSE 量级
    style_base(ax, 'Component (solid: recursive; dashed: iterative)', 'Drift')
    ax.set_ylabel('PCC', fontsize=10.5, fontweight='bold')
    ax2.set_ylabel(r'$\log_{10}$(MSE)', fontsize=10.5, fontweight='bold', color=SINGLE_MSE)


def hp_panel(ax, hp, configs, labels, title, xlabel):
    cfg = hp.set_index('config')
    x = np.arange(len(configs), dtype=float)
    pcc = [float(cfg.loc[c, 'pcc']) for c in configs]
    mse = [np.log10(float(cfg.loc[c, 'mse_norm'])) for c in configs]
    ax.bar(x, pcc, width=0.62, color=SINGLE_PCC, alpha=0.95,
           edgecolor='black', linewidth=0.3)
    annotate_bars(ax, x, pcc)
    ax2 = twin_axis(ax)
    ax2.plot(x, mse, color=SINGLE_MSE, marker='o', ms=3.8, lw=1.6)
    annotate_line(ax2, x, mse, color=SINGLE_MSE, dy=3, fmt='{:.2f}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax2.set_ylim(MSE_LO, MSE_HI); ax2.set_yticks(MSE_TICKS)
    style_base(ax, title, xlabel)
    ax.set_ylabel('PCC', fontsize=10.5, fontweight='bold')
    ax2.set_ylabel(r'$\log_{10}$(MSE)', fontsize=10.5, fontweight='bold', color=SINGLE_MSE)


def main():
    comp = pd.read_csv(COMP)
    hp = pd.read_csv(HP)

    dim_cfgs = ['dim10', 'dim20', 'dim30', 'dim40', 'base', 'dim80', 'dim100', 'dim200']   # dim0(no-GRM 对照) 不在当前网格
    dim_labs = ['10', '20', '30', '40', '50', '80', '100', '200']
    hid_cfgs = ['hid32', 'hid64', 'base', 'hid256']
    hid_labs = ['32', '64', '128', '256']
    dep_cfgs = ['base', 'depth2', 'depth3']
    dep_labs = ['1', '2', '3']
    act_cfgs = ['base', 'act_gelu', 'act_relu', 'act_linear']
    act_labs = ['tanh', 'gelu', 'relu', 'linear']

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.2))

    component_panel(axes[0, 0], comp)
    hp_panel(axes[0, 1], hp, dim_cfgs, dim_labs, 'GRM PCA dim', 'dim')
    hp_panel(axes[0, 2], hp, hid_cfgs, hid_labs, 'Hypernetwork width', 'width')
    hp_panel(axes[1, 0], hp, dep_cfgs, dep_labs, 'Depth', 'depth')
    hp_panel(axes[1, 1], hp, act_cfgs, act_labs, 'Activation', 'activation')
    # 2026-09-22: 删除原 f 面板 (Random seed) —— 主方法已是 5 折内折集成,
    #   随机种子稳定性不再需要额外面板佐证, 故留空该格
    axes[1, 2].axis('off')

    for letter, ax in zip('abcde', axes.ravel()[:5]):
        panel_letter(ax, letter)

    legend_handles = [
        Patch(facecolor=SINGLE_PCC, edgecolor='black', label='PCC (bar)'),
        Line2D([0], [0], color=SINGLE_MSE, marker='o', lw=1.6, label='MSE (line)'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=2,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, bbox_inches='tight')
    print('saved', OUT)


if __name__ == '__main__':
    main()