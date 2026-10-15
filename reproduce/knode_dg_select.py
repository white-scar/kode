#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""knode_dg_select.py -- 为每个数据集选 GRM PCA 维度 D_G（只读现成产物，不训练）。

锁定顺序（2026-09-19 用户方案，本脚本只做第 3 步）:
    0 data freeze -> 1 Full Kode 定义 -> 2 D_G 选择 <- 本脚本
    -> 3 固定 D_G 跑 rank 分析(structural) -> 4 锁架构
    -> 5 主结果十折 -> 6 cond ladder -> 7 外推/基线/统计 -> 8 其他敏感性

口径（全部固定，不在此脚本里改）:
    solver = euler, loss = mse(整轨迹), 评估不含 t1, inner-5 集成, 10 fold-set, seed = 42

选 D_G 的依据只看 **验证集**:
    主方法输出 all_summary 里的 split == 'train' 行就是验证口径 ——
    主方法对每个 fold-set 的 fold1-5 做 inner-5 折, 每个基因型都由「没见过它」的
    inner 模型预测（见 knode_arab_np_tp1_affine.py:247-277 的 val_pos_accum/pred_raw_tr）。
    split == 'test' 是 fold6，只打印出来做参考，**不参与选择**。

预设规则（先定后跑，禁止事后调）:
    1) Pmax = 各 D 里验证 PCC 最高值; Mmin = 验证 MSE 最低值
    2) 合格集 E = { D : pcc_val >= Pmax - GAP  且  mse_val <= Mmin * (1 + WORSE) }
    3) 取 E 中 **最小** 的 D（平台内最紧凑），而不是 test 最高的那个 D

输入（由现有脚本产生，本脚本不重跑）:
    outputs/kode_result/<OUTDIR>/<PREFIX>all_summary_hp_dim{D}.csv
    来源: python knode_hyperparam_ablation.py --dataset <ds> --config dim{D}

输出:
    outputs/kode_result/_dg_select/dg_select_<ds>.csv   每个 D 的验证/测试指标 + eligible 标记
    outputs/kode_result/_dg_select/config_<ds>.json     选中的 D_G 与全部锁定项

用法:
    python knode_dg_select.py --dataset arab
    python knode_dg_select.py --dataset arab --dims 10,20,30,50,80
    python knode_dg_select.py --dataset rice --gap 0.005 --mse-worsen 0.05
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# ---------------- 位置无关: 向上找同时有 data/ 与 outputs/ 的那一级 ----------------
def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").is_dir() and (p / "outputs").is_dir():
            return p
    raise SystemExit("找不到 knode 根目录（需要同时存在 data/ 与 outputs/）")


HERE = Path(__file__).resolve().parent
ROOT = find_root(HERE)

# ---------------- 常量表: 与 knode_hyperparam_ablation.py 保持一致 ----------------
DATASETS = {
    "maize": dict(prefix="knode_tp1_",       outdir="maize",               default_dg=50, dims=[10, 20, 30, 50, 80]),
    "arab":  dict(prefix="knode_arab_tp1_",  outdir="arab",                default_dg=30, dims=[10, 20, 30, 50, 80]),
    "rice":  dict(prefix="knode_rice_tp1_",  outdir="rice_indica_control", default_dg=20, dims=[10, 20, 30, 40, 50]),
    "uav":   dict(prefix="knode_maize_rgb_tp1_", outdir="maize_rgb_drought", default_dg=50, dims=[10, 20, 30, 50, 80]),
}

# ---------------- 锁定项: 写进 config.json, 供后面所有步骤引用 ----------------
LOCKED = {
    "condition": "full",          # Full Kode = [GRM PCA, TP1]
    "rank": 2,                    # 主数据集 MAGIC 的结构分析确定后, 其他数据集固定不调
    "hidden": 128,
    "drift": "affine",
    "solver": "euler",
    "loss": "mse_full_trajectory",
    "eval_excludes_t1": True,
    "inner_folds": 5,
    "foldsets": 10,
    "seed": 42,
    "selection_split": "train(=inner-5 validation)",   # 选超参只看它
}


def load_summary(ds: str, dim: int) -> pd.DataFrame | None:
    info = DATASETS[ds]
    f = ROOT / "outputs" / "kode_result" / info["outdir"] / f"{info['prefix']}all_summary_hp_dim{dim}.csv"
    if not f.exists():
        print(f"  [missing] {f.relative_to(ROOT)}")
        return None
    return pd.read_csv(f)


def pick_row(df: pd.DataFrame, mode: str, split: str):
    sub = df[(df["mode"] == mode) & (df["split"] == split)]
    if sub.empty:
        return None
    return sub.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--dims", default=None, help="逗号分隔, 覆盖默认 D_G 网格")
    ap.add_argument("--mode", default="ode_hyper")
    ap.add_argument("--gap", type=float, default=0.005, help="验证 PCC 允许比最高值低多少")
    ap.add_argument("--mse-worsen", type=float, default=0.05, help="验证 MSE 允许比最优恶化多少(比例)")
    ap.add_argument("--outdir", default=None, help="覆盖输出目录(默认 outputs/kode_result/_dg_select)")
    a = ap.parse_args()

    info = DATASETS[a.dataset]
    dims = [int(x) for x in a.dims.split(",")] if a.dims else info["dims"]
    outdir = Path(a.outdir) if a.outdir else ROOT / "outputs" / "kode_result" / "_dg_select"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"dataset = {a.dataset}   网格 D_G = {dims}   mode = {a.mode}")
    print(f"选择口径 = 验证集 (all_summary 的 split=='train' 行); test 只做参考, 不参与选择")
    print(f"规则: 验证PCC >= Pmax-{a.gap}  且  验证MSE <= Mmin*{1 + a.mse_worsen:.3f}, 在其中取最小 D_G")
    print("=" * 100)

    rows = []
    for d in dims:
        df = load_summary(a.dataset, d)
        if df is None:
            continue
        rv = pick_row(df, a.mode, "train")
        rt = pick_row(df, a.mode, "test")
        if rv is None:
            print(f"  [warn] dim{d}: 找不到 mode={a.mode} / split=train 的行")
            continue
        rows.append(dict(
            dataset=a.dataset, dg=d,
            pcc_val=float(rv.get("pcc", float("nan"))),
            mse_val=float(rv.get("MSE_norm", float("nan"))),
            nrmse_val=float(rv.get("NRMSE", float("nan"))),
            r2_val=float(rv.get("R2_median", float("nan"))),
            pcc_test=float(rt.get("pcc", float("nan"))) if rt is not None else float("nan"),
            mse_test=float(rt.get("MSE_norm", float("nan"))) if rt is not None else float("nan"),
        ))

    if not rows:
        raise SystemExit("没有任何可用的 all_summary_hp_dim*.csv —— 先用 knode_hyperparam_ablation.py 跑 dim 网格")

    t = pd.DataFrame(rows).sort_values("dg").reset_index(drop=True)
    pmax = t["pcc_val"].max()
    mmin = t["mse_val"].min()
    t["eligible"] = (t["pcc_val"] >= pmax - a.gap) & (t["mse_val"] <= mmin * (1.0 + a.mse_worsen))
    t["is_default"] = (t["dg"] == info["default_dg"])

    chain = t[t["eligible"]]
    chosen = int(chain["dg"].min()) if not chain.empty else None

    print()
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"验证 PCC 最高 = {pmax:.4f}   验证 MSE 最低 = {mmin:.6f}")
    print(f"合格 D_G 集合 = {sorted(chain['dg'].tolist())}")
    print(f"==> 选中 D_G = {chosen}  (合格集里最紧凑的一个; 原默认值 = {info['default_dg']})")
    if chosen is not None and chosen == info["default_dg"]:
        print("    与现用默认值一致 —— 不需要改动下游任何常量。")
    elif chosen is not None:
        print(f"    与现用默认值不同 —— 下游所有引用该数据集 D_G 的脚本都要同步改成 {chosen}。")
    print("    (test 列仅作参考: " + ", ".join(f"dg{int(r.dg)}->{r.pcc_test:.4f}" for r in t.itertuples()) + ")")

    csv_p = outdir / f"dg_select_{a.dataset}.csv"
    t.to_csv(csv_p, index=False)
    cfg = dict(LOCKED)
    cfg.update(dict(dataset=a.dataset, dg=chosen, dg_grid=dims,
                    rule=dict(gap=a.gap, mse_worsen=a.mse_worsen, take="smallest eligible D_G"),
                    pcc_val_max=float(pmax), mse_val_min=float(mmin),
                    eligible_dims=sorted(int(x) for x in chain["dg"].tolist()),
                    mode=a.mode,
                    source_files=[f"outputs/kode_result/{info['outdir']}/{info['prefix']}all_summary_hp_dim{d}.csv" for d in dims]))
    json_p = outdir / f"config_{a.dataset}.json"
    json_p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"saved -> {csv_p.relative_to(ROOT)}")
    print(f"saved -> {json_p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
