#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""knode_maize_dg_rank_joint.py -- 玉米 MAGIC Full Kode 的 (D_G x rank) 联合敏感性分析。

================================================================================
【这个脚本跑的是哪个数据集】  <-- 先看这一段
================================================================================
数据目录: data/maize_magic/
  表型 : maize_trait_BLUP_50_NArmv_traits_afterClustering.csv
         long 表, 列为 [bio_ID, accession, DAS, 50 个性状]
         (去掉冗余性状后的 50 个冠层 RGB / 荧光 / 几何 / 纹理描述符 BLUP)
  基因型: GRM_330x330_methodA_from552.csv   330 x 330 (首列 genotype)
  折叠 : cross_validation_folds_w_validation_set_maize.csv   330 x 10, 取值 1..6
                                 (折内: 1-5 = 训练/早停, 6 = 外层测试集)

规模 (已在 load_base() 里硬检查):
  N = 330 个品系
  T = 25 个时间点 (DAS 15,16,17,18,19,22,23,24,25,26,29,30,31,32,33,36,37,38,39,40,
                    43,44,45,46,47; 间隔 1-4 天, 不规则)
  P = 50 个性状

玉米 MAGIC 只有一套表型 / 基因型 / 折划分 (没有环境分层), 所以本脚本没有 --env 开关。
本项目玉米 MAGIC 的默认 GRM 输入维度是 D_G = 50 (与主方法 knode_maize_np_tp1_affine.py 一致)。
================================================================================

目的: 完整、无偏地跑完 D_G x rank 网格, **由程序只提供客观结果, 不自动筛参、不挑最优**。
最终保留哪一组 (D_G, rank), 由人看完整表与四个矩阵后判断。

模型主体 (与拟南芥/水稻/RGB 的 Kode 一致, 本次不改):
    c_i = [z_i(D_G), y_i(t1)]  ->  Linear+tanh  ->  U_i, V_i, b_i
    A_i = U_i V_i,  dh_i/dt = h_i A_i^T + b_i          (affine drift 保留)
    训练 loss  = 整轨迹 MSE;  validation/test 指标排除 t1
    normalized real time gaps 上的 explicit Euler; HID=128; Tanh
    Adam lr=1e-3, weight decay=1e-4, grad clip=1.0, 1000 epochs
    5 个 inner model 对 fold6 做 ensemble
    RNG: 每个 (D_G, rank) cell 开始前重置 torch.manual_seed(42) / np.random.seed(42)

缓存隔离 (防止冒烟污染正式):
    outputs/reproduce/dg_rank_maize/ckpt_final/   <- 只有 --run   读写
    outputs/reproduce/dg_rank_maize/ckpt_smoke/   <- 只有 --smoke 读写
    文件名: dg{DG}_r{rank}_f{fold}_i{inner}_affine_seed42_ep{epochs}.npz
    npz 内 meta 记录 dataset/D_G/rank/fold/inner/seed/epochs/hidden/affine;
    resume 时逐项校验, 与当前配置不一致则拒绝加载。

输出 (outputs/reproduce/dg_rank_maize/):
    cells/f{fold}_dg{DG}_r{rank}_ep{epochs}.csv
                           每个 cell 一个独立文件 (含 val/test x pcc/mse + dataset/D_G/rank/fold/epochs)
                           **一个文件只由一个进程写; 文件已存在就跳过, 绝不重算/绝不覆盖**
    agg.csv                汇总的完整二维表: val_pcc_mean/std, val_mse_mean/std,
                           test_pcc_mean/std, test_mse_mean/std, n_foldsets
    matrix_val_pcc.csv / matrix_val_mse.csv / matrix_test_pcc.csv / matrix_test_mse.csv
                           D_G x rank 四个矩阵
    dataset_manifest.json  数据身份 (文件名/N/T/P/25个DAS/50个性状名)
    run_meta.json          本次运行的协议与网格 (只登记, 不含任何选择结论)
    smoke_cells/           --smoke 的 cell 结果 (永远不会被 aggregate 读到)

不覆盖规则:
    1) cell 结果按 (fold, D_G, rank, epochs) 命名, 已存在 => 跳过, 后来跑的永远覆盖不了先跑的;
    2) 每个 cell 一个文件, 40 个进程并行时各写各的, 不存在并发覆盖;
    3) 落盘用 tmp + rename, 不会留下写坏的半截文件;
    4) aggregate 只做派生: 汇总前先检查 cells/ 内 epochs 唯一、(fold,D_G,rank) 无重复, 不满足直接报错;
    5) 想重算某个 cell: 删掉它那个 cells/f{f}_dg{dg}_r{rank}_ep{ep}.csv (和对应 ckpt) 再跑。

用法:
    python knode_maize_dg_rank_joint.py                 # 只自检(不训练)
    python knode_maize_dg_rank_joint.py --run --smoke   # 冒烟: 1 fold x 1 cell x 50 ep, 走 ckpt_smoke
    python knode_maize_dg_rank_joint.py --run           # 全量: 10 fold x 24 cell, 走 ckpt_final
    python knode_maize_dg_rank_joint.py --aggregate-only
"""
import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ---------------- 位置无关: 向上找同时有 data/ 与 outputs/ 的那一级 ----------------
def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").is_dir() and (p / "outputs").is_dir():
            return p
    raise SystemExit("找不到 knode 根目录 (需要同时存在 data/ 与 outputs/)")


HERE = Path(__file__).resolve().parent
ROOT = find_root(HERE)

# ---------------- 协议常量 (模型主体不改) ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3
WD = 1e-4
HID = 128
SEED = 42
DATASET = "maize"
DATA_SUB = "maize_magic"

DG_GRID = [10, 20, 30, 40, 50, 80]
RANK_GRID = [1, 2, 3, 4]

# 玉米 MAGIC 正式数据的身份 (硬检查)
EXP_N, EXP_T, EXP_P = 330, 25, 50

PHENO_FILE = "maize_trait_BLUP_50_NArmv_traits_afterClustering.csv"
GRM_FILE = "GRM_330x330_methodA_from552.csv"
CV_FILE = "cross_validation_folds_w_validation_set_maize.csv"

RUN_DIR = ROOT / "outputs" / "reproduce" / "dg_rank_maize"
CKPT_FINAL = RUN_DIR / "ckpt_final"
CKPT_SMOKE = RUN_DIR / "ckpt_smoke"

PROTOCOL = {
    "dataset": DATASET,
    "condition": "full",
    "drift": "affine",
    "affine_drift": True,
    "hidden": HID,
    "activation": "tanh",
    "optimizer": "adam",
    "lr": LR,
    "weight_decay": WD,
    "grad_clip": 1.0,
    "solver": "explicit_euler_normalized_real_gaps",
    "loss": "mse_full_trajectory",
    "checkpoint_metric": "validation_mse_excluding_t1",
    "inner_folds": 5,
    "outer_foldsets": 10,
    "seed": SEED,
    "ensemble": "5 inner models averaged on fold6",
    "eval_excludes_t1": True,
    "rng": "reset torch/np seed at the start of every (D_G, rank) cell",
    "note": "本脚本只输出结果, 不自动筛参; D_G*/rank* 由人工判断",
}


# ---------------- 数据 ----------------
P = T = N = None
Y_all = times_all = trait_cols = cv_foldsets = None
times_t = None
grm_raw_full = None


def load_base():
    """加载玉米 MAGIC 数据; 硬检查 N/T/P 并在第一次加载时写出 dataset_manifest.json."""
    global P, T, N, Y_all, times_all, trait_cols, cv_foldsets, times_t, grm_raw_full
    pheno = pd.read_csv(ROOT / "data" / DATA_SUB / PHENO_FILE)
    cv_raw = pd.read_csv(ROOT / "data" / DATA_SUB / CV_FILE, index_col=0)
    grm_full = pd.read_csv(ROOT / "data" / DATA_SUB / GRM_FILE, index_col=0)

    meta_cols = ["bio_ID", "accession", "DAS"]
    trait_cols = [c for c in pheno.columns if c not in meta_cols and not c.startswith("Unnamed") and c != ""]
    P = len(trait_cols)
    times_all = np.array(sorted(pheno["DAS"].unique()), dtype=np.float64)
    T = len(times_all)
    common_ids = sorted(set(pheno["bio_ID"].astype(str)) & set(cv_raw.index))
    N = len(common_ids)

    assert P == EXP_P, f"Unexpected trait number: P={P}, expected {EXP_P}"
    assert T == EXP_T, f"Unexpected time points: T={T}, expected {EXP_T}"
    assert N == EXP_N, f"Unexpected genotype number: N={N}, expected {EXP_N}"

    cv_foldsets = cv_raw.loc[common_ids].iloc[:, :].values.astype(int)
    grm_full.index = grm_full.index.astype(str)
    grm_full.columns = grm_full.columns.astype(str)
    grm_raw_full = grm_full.loc[common_ids, common_ids].values.astype(np.float64)

    Y_all = np.full((N, T, P), np.nan, dtype=np.float64)
    for pi, trait in enumerate(trait_cols):
        sub = pheno[["bio_ID", "DAS", trait]].copy()
        sub["bio_ID"] = sub["bio_ID"].astype(str)
        Y_all[:, :, pi] = sub.pivot(index="bio_ID", columns="DAS", values=trait).loc[common_ids].values
    assert not np.any(np.isnan(Y_all)), "表型矩阵含 NaN"

    tn = (times_all - times_all.min()) / (times_all.max() - times_all.min() + 1e-8)
    times_t = torch.tensor(tn, dtype=torch.float32, device=DEVICE)

    print(f"[data] N={N} T={T} P={P}  GRM={grm_raw_full.shape[0]}x{grm_raw_full.shape[1]}  "
          f"CV={cv_foldsets.shape[1]} fold-set")
    print(f"[data] 硬检查通过: N={EXP_N} T={EXP_T} P={EXP_P}")

    manifest = dict(dataset=DATASET,
                    phenotype_file=str(Path("data") / DATA_SUB / PHENO_FILE),
                    grm_file=str(Path("data") / DATA_SUB / GRM_FILE),
                    cv_file=str(Path("data") / DATA_SUB / CV_FILE),
                    N=int(N), T=int(T), P=int(P),
                    DAS=[float(x) for x in times_all],
                    trait_names=list(trait_cols),
                    cv_foldsets=int(cv_foldsets.shape[1]))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[data] saved -> {(RUN_DIR / 'dataset_manifest.json')}")


def pca_for(dg: int) -> np.ndarray:
    """按当前 cell 的 D_G 现算 PCA (不用任何全局维度常量)."""
    return PCA(n_components=int(dg)).fit_transform(grm_raw_full).astype(np.float32)


def fold_split(fold: int):
    fc = cv_foldsets[:, fold]
    in_tr = np.where((fc >= 1) & (fc <= 5))[0]
    in_val = np.where(fc == 6)[0]
    return in_tr, in_val, fc


# ---------------- 模型 ----------------
class MaskedHyperODEModel(nn.Module):
    """dh/dt = h (U V)^T + b, U,V,b 由超网从 [grm_pca(D_G), y1] 生成."""

    def __init__(self, P, rank, grm_dim, hid=HID):
        super().__init__()
        self.P = int(P)
        self.rank = int(rank)
        self.grm_dim = int(grm_dim)
        self.cond_in = self.grm_dim + self.P
        self.hyper_enc = nn.Sequential(nn.Linear(self.cond_in, hid), nn.Tanh())
        self.hyper_U = nn.Linear(hid, self.P * self.rank)
        self.hyper_V = nn.Linear(hid, self.rank * self.P)
        self.hyper_b = nn.Linear(hid, self.P)
        self._U_dyn = self._V_dyn = self._b_dyn = None

    def set_z(self, g, t):
        if g is None:
            g = torch.zeros(t.shape[0], self.grm_dim, device=DEVICE)
        if t is None:
            t = torch.zeros(g.shape[0], self.P, device=DEVICE)
        cond = torch.cat([g, t], dim=-1)
        B = cond.shape[0]
        h = self.hyper_enc(cond)
        self._U_dyn = self.hyper_U(h).view(B, self.P, self.rank)
        self._V_dyn = self.hyper_V(h).view(B, self.rank, self.P)
        self._b_dyn = self.hyper_b(h).view(B, self.P)

    def forward(self, t, h):
        lin = torch.bmm(torch.bmm(h.unsqueeze(1), self._V_dyn.transpose(1, 2)),
                        self._U_dyn.transpose(1, 2)).squeeze(1)
        return lin + self._b_dyn


def euler_integrate(model, h0, ts):
    B, D = h0.shape
    L = len(ts)
    traj = torch.zeros(B, L, D, device=h0.device, dtype=h0.dtype)
    traj[:, 0] = h0
    h = h0
    for i in range(1, L):
        dt = max(ts[i].item() - ts[i - 1].item(), 1e-6)
        h = h + model(None, h) * dt
        traj[:, i] = h
    return traj


def channels(ids, grm_dim, grm_pca, y1):
    ids = np.asarray(ids)
    return torch.tensor(grm_pca[ids], device=DEVICE), torch.tensor(y1[ids], device=DEVICE)


def train_model(rank, grm_dim, grm_pca, y1, Ys_tr, tr_ids, Ys_ea, ea_ids, n_epochs):
    m = MaskedHyperODEModel(P, rank, grm_dim, HID).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    y_tr = torch.tensor(Ys_tr.astype(np.float32), device=DEVICE)
    g_tr, t_tr = channels(tr_ids, grm_dim, grm_pca, y1)
    y_ea = torch.tensor(Ys_ea.astype(np.float32), device=DEVICE)
    g_ea, t_ea = channels(ea_ids, grm_dim, grm_pca, y1)
    best = None
    best_loss = float("inf")
    for ep in range(n_epochs):
        m.train()
        m.set_z(g_tr, t_tr)
        pred = euler_integrate(m, y_tr[:, 0, :], times_t)
        loss = F.mse_loss(pred, y_tr)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if ep % 50 == 0 or ep == n_epochs - 1:
            with torch.no_grad():
                m.eval()
                m.set_z(g_ea, t_ea)
                pv = euler_integrate(m, y_ea[:, 0, :], times_t)
                vl = F.mse_loss(pv[:, 1:, :], y_ea[:, 1:, :]).item()
            if best is None or (not math.isnan(vl) and vl < best_loss):
                best_loss = vl
                best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best)
    m.eval()
    return m


def predict_ids(m, grm_dim, grm_pca, y1, ids):
    """模型自身 rollout (含 b_i)."""
    g, t = channels(ids, grm_dim, grm_pca, y1)
    with torch.no_grad():
        m.set_z(g, t)
        h0 = torch.tensor(y1[np.asarray(ids)], device=DEVICE)
        out = euler_integrate(m, h0, times_t)
    return out.cpu().numpy()


def pcc_mse(pred, Y_true):
    """逐时点跨性状平均 PCC (t>=1) + t>=1 的 MSE, 均在训练折归一化空间."""
    pccs = []
    for t in range(1, T):
        rs = []
        for p in range(P):
            if np.std(Y_true[:, t, p]) < 1e-12:
                continue
            rs.append(pearsonr(pred[:, t, p], Y_true[:, t, p])[0])
        pccs.append(float(np.mean(rs)) if rs else float("nan"))
    mse = float(np.mean((pred[:, 1:, :] - Y_true[:, 1:, :]) ** 2))
    return float(np.nanmean(pccs)), mse


# ---------------- 单个 cell ----------------
def cell_meta(dg, rank, fold, inner, epochs):
    return dict(dataset=DATASET, D_G=int(dg), rank=int(rank), fold=int(fold), inner=int(inner),
                seed=SEED, epochs=int(epochs), hidden=HID, affine=True)


def ckpt_path(ckpt_dir: Path, dg, rank, fold, inner, epochs) -> Path:
    return ckpt_dir / f"dg{dg}_r{rank}_f{fold}_i{inner}_affine_seed{SEED}_ep{epochs}.npz"


def load_ckpt(path: Path, want: dict):
    """resume: meta 与当前配置逐项一致才允许读取, 否则拒绝."""
    z = np.load(path, allow_pickle=True)
    if "meta" not in z:
        raise RuntimeError(f"{path.name} 缺少 meta, 拒绝加载 (旧格式缓存)")
    got = json.loads(str(z["meta"][0]))
    diff = {k: (got.get(k), want[k]) for k in want if got.get(k) != want[k]}
    if diff:
        raise RuntimeError(f"{path.name} meta 与当前配置不一致 {diff}, 拒绝加载")
    return z["pred_val"], z["pred_test"]


def run_cell(dg, rank, fold, n_epochs, ckpt_dir):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    in_tr, in_val, fc = fold_split(fold)
    tr_fc = fc[in_tr]
    grm_pca = pca_for(dg)

    tr_min = np.nanmin(Y_all[in_tr], axis=(0, 1))
    tr_max = np.nanmax(Y_all[in_tr], axis=(0, 1))
    tr_rng = np.maximum(tr_max - tr_min, 1e-8)
    Ys_all = ((Y_all - tr_min) / tr_rng).astype(np.float32)
    y1 = Ys_all[:, 0, :]

    valid_pred = np.zeros((len(in_tr), T, P), dtype=np.float64)
    test_preds = []
    for inner in range(1, 6):
        itr = in_tr[tr_fc != inner]
        iva = in_tr[tr_fc == inner]
        ck = ckpt_path(ckpt_dir, dg, rank, fold, inner, n_epochs)
        if ck.exists():
            pv_val, pv_test = load_ckpt(ck, cell_meta(dg, rank, fold, inner, n_epochs))
        else:
            t0 = time.time()
            m = train_model(rank, dg, grm_pca, y1, Ys_all[itr], itr, Ys_all[iva], iva, n_epochs)
            pv_val = predict_ids(m, dg, grm_pca, y1, iva)
            pv_test = predict_ids(m, dg, grm_pca, y1, in_val)
            np.savez_compressed(
                ck, pred_val=pv_val.astype(np.float32), pred_test=pv_test.astype(np.float32),
                meta=np.array([json.dumps(cell_meta(dg, rank, fold, inner, n_epochs))], dtype=object))
            print(f"    [train dg{dg}/r{rank}/i{inner}] {time.time() - t0:.1f}s", flush=True)
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        valid_pred[tr_fc == inner] = pv_val
        test_preds.append(pv_test)

    test_pred = np.mean(np.stack(test_preds), axis=0)
    val_pcc, val_mse = pcc_mse(valid_pred, Ys_all[in_tr])
    te_pcc, te_mse = pcc_mse(test_pred, Ys_all[in_val])
    return dict(dataset=DATASET, D_G=int(dg), rank=int(rank), fold=int(fold),
                epochs=int(n_epochs),
                val_pcc=round(val_pcc, 6), val_mse=round(val_mse, 6),
                test_pcc=round(te_pcc, 6), test_mse=round(te_mse, 6),
                n_train=int(len(in_tr)), n_test=int(len(in_val)))


def cell_dir(smoke: bool) -> Path:
    d = RUN_DIR / ("smoke_cells" if smoke else "cells")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cell_files(smoke: bool):
    d = RUN_DIR / ("smoke_cells" if smoke else "cells")
    return sorted(d.glob("*.csv")) if d.is_dir() else []


def cell_csv_path(fold, dg, rank, epochs, smoke: bool) -> Path:
    """每个 (fold, D_G, rank, epochs) 一个独立文件: 各进程各写各的, 不存在并发覆盖。"""
    return cell_dir(smoke) / f"f{fold}_dg{dg}_r{rank}_ep{epochs}.csv"


def run_fold(fold, dgs, ranks, n_epochs, ckpt_dir, smoke):
    """逐个 cell 跑, 每个 cell 单独落盘。

    该 cell 的 csv 已存在 => 读出来直接跳过 (绝不重算、绝不覆盖)。
    要重算: 先删掉 cells/f{fold}_dg{dg}_r{rank}_ep{epochs}.csv (和对应 ckpt) 再跑。
    """
    rows = []
    for dg in dgs:
        for rank in ranks:
            p = cell_csv_path(fold, dg, rank, n_epochs, smoke)
            if p.exists():
                row = pd.read_csv(p).to_dict("records")[0]
                rows.append(row)
                print(f"  [dg{dg}/r{rank}/f{fold}] 已有结果, 跳过不覆盖 -> {p.name}", flush=True)
                continue
            row = run_cell(dg, rank, fold, n_epochs, ckpt_dir)
            tmp = p.with_suffix(".csv.tmp")
            pd.DataFrame([row]).to_csv(tmp, index=False)
            tmp.replace(p)      # 原子落盘: 不会留下写坏的半截文件
            rows.append(row)
            print(f"  [dg{dg}/r{rank}/f{fold}] val_pcc={row['val_pcc']:.4f} val_mse={row['val_mse']:.6f} "
                  f"| test_pcc={row['test_pcc']:.4f} test_mse={row['test_mse']:.6f}", flush=True)
    return rows


# ---------------- 汇总: 只出表与矩阵, 不做任何选择 ----------------
def aggregate(smoke=False, dgs=None, ranks=None):
    dgs = dgs or DG_GRID
    ranks = ranks or RANK_GRID
    files = _cell_files(smoke)
    if not files:
        legacy = sorted(RUN_DIR.glob("smoke_f[0-9].csv" if smoke else "f[0-9].csv"))
        if legacy:
            print("[warn] 只找到旧格式 (逐折) f*.csv, 按旧格式汇总")
            files = legacy
    if not files:
        print("没有可汇总的 f*.csv, 先 --run")
        return None
    df = pd.concat([pd.read_csv(f) for f in files])
    for col in ("fold", "D_G", "rank", "epochs"):
        assert col in df.columns, f"cell CSV 缺少列 {col}"
    eps = sorted(set(int(x) for x in df["epochs"]))
    assert len(eps) == 1, f"cells/ 里混了多个 epochs={eps}, 拒绝汇总 (避免混口径)"
    dup = int(df.duplicated(subset=["fold", "D_G", "rank"]).sum())
    assert dup == 0, f"有 {dup} 条重复的 (fold, D_G, rank), 拒绝汇总"
    print(f"汇总 {len(files)} 个 cell 文件 | epochs={eps[0]} | folds={sorted(set(int(x) for x in df['fold']))}")
    g = df.groupby(["D_G", "rank"]).agg(
        val_pcc_mean=("val_pcc", "mean"), val_pcc_std=("val_pcc", "std"),
        val_mse_mean=("val_mse", "mean"), val_mse_std=("val_mse", "std"),
        test_pcc_mean=("test_pcc", "mean"), test_pcc_std=("test_pcc", "std"),
        test_mse_mean=("test_mse", "mean"), test_mse_std=("test_mse", "std"),
        n_foldsets=("val_pcc", "count")).reset_index()

    tag = "smoke_agg.csv" if smoke else "agg.csv"
    out = RUN_DIR / tag
    g.to_csv(out, index=False)

    mats = {}
    for name, col in [("matrix_val_pcc", "val_pcc_mean"), ("matrix_val_mse", "val_mse_mean"),
                      ("matrix_test_pcc", "test_pcc_mean"), ("matrix_test_mse", "test_mse_mean")]:
        m = g.pivot(index="D_G", columns="rank", values=col).reindex(index=dgs, columns=ranks).round(6)
        mp = RUN_DIR / (f"{'smoke_' if smoke else ''}{name}.csv")
        m.to_csv(mp)
        mats[name] = m

    print()
    print("=" * 108)
    print("完整结果表 (每个 (D_G, rank) 一行) —— 程序不做任何筛选, 最终参数由人工判断")
    print("=" * 108)
    print(g.sort_values(["D_G", "rank"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for name in ["matrix_val_pcc", "matrix_val_mse", "matrix_test_pcc", "matrix_test_mse"]:
        print()
        print(f"--- {name}  (行=D_G, 列=rank) ---")
        print(mats[name].to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"saved -> {out}  (含 val_pcc_mean/std, val_mse_mean/std, "
          f"test_pcc_mean/std, test_mse_mean/std, n_foldsets)")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="真正训练 (默认只自检)")
    ap.add_argument("--smoke", action="store_true", help="冒烟: 1 fold x 1 cell x 50 ep, 只走 ckpt_smoke")
    ap.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--dg", default=None, help="逗号分隔, 默认 10,20,30,40,50")
    ap.add_argument("--ranks", default=None, help="逗号分隔, 默认 1,2,3,4")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--aggregate-only", action="store_true", help="只用已有 CSV 重算表与矩阵")
    a = ap.parse_args()

    dgs = [int(x) for x in a.dg.split(",")] if a.dg else DG_GRID
    ranks = [int(x) for x in a.ranks.split(",")] if a.ranks else RANK_GRID
    folds = [int(x) for x in a.folds.split(",") if x.strip()]
    epochs = 50 if a.smoke else a.epochs
    if not a.smoke and a.epochs != 1000:
        raise SystemExit("正式结果只允许 epochs=1000 (防短跑污染正式结果); 短跑请加 --smoke")
    if a.smoke:
        folds = folds[:1]
        dgs = dgs[:1]
        ranks = ranks[:1]

    ckpt_dir = CKPT_SMOKE if a.smoke else CKPT_FINAL
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if a.aggregate_only:
        aggregate(smoke=a.smoke, dgs=dgs, ranks=ranks)
        return

    print(f"Device={DEVICE}  root={ROOT}")
    print(f"网格: D_G={dgs} x rank={ranks}  folds={folds}  epochs={epochs}")
    print(f"条件: full  |  drift: affine (含 b_i)  |  seed: {SEED}")
    print(f"缓存: {ckpt_dir}   ({'冒烟专用' if a.smoke else '正式专用'}, 两者不共享)")
    print(f"结果: {(RUN_DIR / ('smoke_cells' if a.smoke else 'cells'))}")
    print("本脚本不自动筛参: 只输出完整表 + 四个矩阵 (val/test x pcc/mse)")
    print("-" * 108)

    if not a.run:
        print("[dry] 未加 --run, 不训练。")
        print(f"      网格 = {len(dgs)} x {len(ranks)} = {len(dgs) * len(ranks)} 个 cell, "
              f"每个 cell = {len(folds)} fold x 5 inner")
        print("      先冒烟: --run --smoke")
        return

    if not a.smoke:
        (RUN_DIR / "run_meta.json").write_text(json.dumps(
            {**PROTOCOL, "grid": dict(D_G=dgs, rank=ranks), "folds": folds, "epochs": epochs},
            indent=2, ensure_ascii=False), encoding="utf-8")

    load_base()
    for fold in folds:
        print(f"\n===== fold-set {fold} =====")
        run_fold(fold, dgs, ranks, epochs, ckpt_dir, a.smoke)
    aggregate(smoke=a.smoke, dgs=dgs, ranks=ranks)


if __name__ == "__main__":
    main()
