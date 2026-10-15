#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified current-protocol hyperparameter ablation runner.

This is the replacement for the deleted old maize/arab architecture-ablation
scripts. It imports the current main method script and only changes the model
capacity / encoder configuration, so split, normalization, inner-5 ensemble,
ODE and affine logic stay identical.

Supported sweeps:
  dim   : GRM PCA dimension
  hid   : hypernetwork width
  depth : hypernetwork depth
  act   : activation
  seed  : random seed

Single config:
  python knode_hyperparam_ablation.py --dataset maize --config hid32

All configs:
  python knode_hyperparam_ablation.py --dataset maize --run-all

List configs:
  python knode_hyperparam_ablation.py --dataset maize --list
"""
import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
REPRO = ROOT / 'reproduce'
PY = Path(sys.executable)

DEFAULT = {
    'maize': dict(grm_dim=50, hid=128, depth=1, act='tanh', seed=42),
    'arab':  dict(grm_dim=30, hid=128, depth=1, act='tanh', seed=42),
    'rice':  dict(grm_dim=10, hid=128, depth=1, act='tanh', seed=42),   # 2026-09-19 锁定 D_G=10
    'uav':   dict(grm_dim=50, hid=128, depth=1, act='tanh', seed=42),
}
DIMS = {
    'maize': [10, 20, 30, 40, 80, 100, 200],
    'arab':  [10, 20, 40, 50],
    'rice':  [5, 20, 30, 50],          # base=10 两侧都扫 (5 更小, 其余更大)
    'uav':   [10, 20, 30, 40, 80, 100],
}
# ???????????? (aggregate ? all_summary{TAG}.csv ?)
PREFIX = {
    'maize': 'knode_tp1_',
    'arab':  'knode_arab_tp1_',
    'rice':  'knode_rice_tp1_',
    'uav':   'knode_maize_rgb_tp1_',
}

# ??????????? (kode_result ?)
OUTDIR = {
    'maize': 'maize',
    'arab':  'arab',
    'rice':  'rice_indica_control',
    'uav':   'maize_rgb_drought',
}

BASE_MODULE = {
    'maize': 'knode_maize_np_tp1_affine',
    'arab':  'knode_arab_np_tp1_affine',
    'rice':  'knode_rice_np_tp1_affine',
    'uav':   'knode_maize_rgb_drought_np_tp1_affine',
}

# 主方法脚本所在的子目录 (搬进子目录后必须手动加 sys.path)
SUBDIR = {
    'maize': REPRO / 'maize_magic',
    'arab':  REPRO / 'arab',
    'rice':  REPRO / 'rice',
    'uav':   REPRO / 'maize_RGB',
}
for _d in SUBDIR.values():
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))


def config_map(dataset):
    base = dict(DEFAULT[dataset])
    cfgs = {'base': dict(base)}
    for d in DIMS[dataset]:
        cfgs[f'dim{d}'] = {**base, 'grm_dim': d}
    for w in [32, 64, 256]:
        cfgs[f'hid{w}'] = {**base, 'hid': w}
    for dep in [2, 3]:
        cfgs[f'depth{dep}'] = {**base, 'depth': dep}
    for act in ['gelu', 'relu', 'linear']:
        cfgs[f'act_{act}'] = {**base, 'act': act}
    for sd in [1234, 2024, 2025]:
        cfgs[f'seed{sd}'] = {**base, 'seed': sd}
    return cfgs


def make_encoder(cond_in, hid, depth, act):
    act_map = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'linear': None,
    }
    if act not in act_map:
        raise ValueError(f'unsupported activation: {act}')
    layers = [nn.Linear(cond_in, hid)]
    act_fn = act_map[act]
    if act_fn is not None:
        layers.append(act_fn())
    for _ in range(depth - 1):
        layers.append(nn.Linear(hid, hid))
        if act_fn is not None:
            layers.append(act_fn())
    return nn.Sequential(*layers)


def make_model_class(base, depth, act):
    class HyperparamHyperODEModel(base.HyperODEModel):
        def __init__(self, P=50, COND_IN=100, rank=2, hid=128, drift='linear'):
            nn.Module.__init__(self)
            self.P = P
            self.rank = rank
            self.drift = drift
            self.hyper_enc = make_encoder(COND_IN, hid, depth, act)
            self.hyper_U = nn.Linear(hid, P * rank)
            self.hyper_V = nn.Linear(hid, rank * P)
            if drift == 'affine':
                self.hyper_b = nn.Linear(hid, P)
            self._U_dyn = self._V_dyn = self._b_dyn = None

        def set_z(self, grm_pca, y1):
            cond = torch.cat([grm_pca, y1], dim=-1)
            B = grm_pca.shape[0]
            h = self.hyper_enc(cond)
            self._U_dyn = self.hyper_U(h).view(B, self.P, self.rank)
            self._V_dyn = self.hyper_V(h).view(B, self.rank, self.P)
            if self.drift == 'affine':
                self._b_dyn = self.hyper_b(h).view(B, self.P)

    return HyperparamHyperODEModel


def run_single(args):
    base = importlib.import_module(BASE_MODULE[args.dataset])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    base.SEED = args.seed
    base.GRM_DIM = args.grm_dim
    base.HID = args.hid
    base.HyperODEModel = make_model_class(base, args.depth, args.act)
    sys.argv = [
        sys.argv[0],
        '--epochs', str(args.epochs),
        '--foldsets', str(args.foldsets),
        '--drift', args.drift,
        '--mode', args.mode,
        '--tag', args.tag,
        '--seed', str(args.seed),
    ]
    print(
        f'[hyperparam] dataset={args.dataset} tag={args.tag} '
        f'grm_dim={args.grm_dim} hid={args.hid} depth={args.depth} '
        f'act={args.act} seed={args.seed} mode={args.mode} drift={args.drift}',
        flush=True,
    )
    base.main()


def cmd_for(dataset, name, cfg, common, tag):
    return [
        str(PY), str(Path(__file__).resolve()),
        '--dataset', dataset,
        '--_single', name,
        '--epochs', str(common.epochs),
        '--foldsets', str(common.foldsets),
        '--drift', common.drift,
        '--mode', common.mode,
        '--grm-dim', str(cfg['grm_dim']),
        '--hid', str(cfg['hid']),
        '--depth', str(cfg['depth']),
        '--act', cfg['act'],
        '--seed', str(cfg['seed']),
        '--tag', tag,
    ]


def aggregate(dataset, names, tags):
    import pandas as pd
    outdir = ROOT / 'outputs' / 'kode_result' / OUTDIR[dataset]
    rows = []
    for name, tag in zip(names, tags):
        f = outdir / f'{PREFIX[dataset]}all_summary{tag}.csv'
        if not f.exists():
            raise FileNotFoundError(f'missing summary: {f}')
        d = pd.read_csv(f)
        d['dataset'] = dataset
        d['config'] = name
        rows.append(d)
    out = ROOT / 'outputs' / 'reproduce' / f'hyperparam_ablation_{dataset}_summary.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    print('Saved ->', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['maize', 'arab', 'rice', 'uav'], required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--run-all', action='store_true')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--foldsets', type=int, default=10)
    ap.add_argument('--drift', choices=['linear', 'affine'], default='affine')
    ap.add_argument('--mode', default='ode_hyper,iter_hyper')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--_single', default=None)
    # Single-run explicit values.
    ap.add_argument('--grm-dim', type=int, default=None)
    ap.add_argument('--hid', type=int, default=None)
    ap.add_argument('--depth', type=int, default=None)
    ap.add_argument('--act', choices=['tanh', 'relu', 'gelu', 'linear'], default=None)
    ap.add_argument('--seed', type=int, default=None)
    a = ap.parse_args()

    cfgs = config_map(a.dataset)
    if a.list:
        for name, cfg in cfgs.items():
            print(f'{name}\t{cfg}')
        return

    if a._single:
        cfg = cfgs[a._single]
        for key in ['grm_dim', 'hid', 'depth', 'act', 'seed']:
            val = getattr(a, key)
            if val is not None:
                cfg[key] = val
        a.grm_dim = cfg['grm_dim']; a.hid = cfg['hid']; a.depth = cfg['depth']
        a.act = cfg['act']; a.seed = cfg['seed']
        if a.tag is None:
            a.tag = f'_hp_{a._single}'
        run_single(a)
        return

    names = [a.config] if a.config else (list(cfgs) if a.run_all else None)
    if not names:
        raise SystemExit('use --config NAME, --run-all, or --list')
    for name in names:
        if name not in cfgs:
            raise SystemExit(f'unknown config: {name}')

    tags = []
    for name in names:
        tag = f'_hp_{name}'
        tags.append(tag)
        cmd = cmd_for(a.dataset, name, cfgs[name], a, tag)
        print('RUN', ' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(REPRO), check=True)
    aggregate(a.dataset, names, tags)


if __name__ == '__main__':
    main()