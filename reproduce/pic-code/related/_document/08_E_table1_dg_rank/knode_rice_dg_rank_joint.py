#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""knode_rice_dg_rank_joint.py -- 水稻 (indica) Full Kode 的 (D_G x rank) 联合敏感性分析。

================================================================================
【这个脚本跑的是哪个数据集 / 两个环境是什么关系】  <-- 先看这一段
================================================================================
数据目录: data/rice_salinity/
  表型 : rice_indica_control_traits.csv   (--env control, 默认; 主图用的就是这个)
         rice_indica_salt_traits.csv      (--env salt)
         long 表, 列为 [Unnamed: 0, bio_ID, accession, DAS, 20 个性状]
  基因型: rice_GRM_226x226.csv   226 x 226 VanRaden GRM (首列 genotype)
  折叠 : rice_cv_10fold.csv      226 x 10 (V1..V10), 取值 1..6 = 折标签
                                 (折内: 1-5 = 训练/早停, 6 = 外层测试集)

规模 (control 与 salt 完全一致, 已在 load_base() 里硬检查):
  N = 226 个品系
  T = 13 个时间点 (DAS 30,31,32,33,34,35,36,37,38,39,40,41,42; 间隔 1-2 天)
  P = 20 个性状 (PSA + SV1_*/SV2_* 系列 RGB 形态描述符)
  GRM 另有 rice_GRM_228x228.csv (多 2 个没有表型的品系), 本项目不使用。

control 与 salt 的关系 (决定能不能做配对比较):
  - 表型文件不同, 但性状列名相同, P 都是 20;
  - **基因型完全相同**: 两个环境共用一个 GRM 文件, 同一批 226 个 accession;
  - **折划分完全相同**: 项目里只有一份 rice_cv_10fold.csv, 两环境共用同一组 V1..V10。
  => 同一群体 + 同一折 + 同一套性状名, 所以 control vs salt 可以配对比较。
  本脚本仍然把两者当成两个独立数据集分别跑、分别输出 (输出目录不同, 不会互相覆盖)。

时间窗: 表型文件本来是 DAS 28 起共 14 个时间点, 但作者 dynamicGP 用的是 DAS 30-42。
        为了让两种方法在同一个时间窗上可比, 本脚本在 load_base() 里直接丢掉 DAS 28
        (DAS_MIN=30, 见 ENV_CFG), 实际用 13 个时间点, 硬检查值 T 也相应改成 13。
================================================================================

目的: 完整、无偏地跑完 D_G x rank 网格, **由程序只提供客观结果, 不自动筛参、不挑最优**。
水稻项目默认 D_G=20 / r=2; 最终保留哪一组, 由人看完整表与四个矩阵后判断。

模型主体 (与玉米/拟南芥 Kode 一致, 本次不改):
    c_i = [z_i(D_G), y_i(t1)]  ->  Linear+tanh  ->  U_i, V_i, b_i
    A_i = U_i V_i,  dh_i/dt = h_i A_i^T + b_i          (affine drift 保留)
    训练 loss  = 整轨迹 MSE;  validation/test 指标排除 t1
    normalized real time gaps 上的 explicit Euler; HID=128; Tanh
    Adam lr=1e-3, weight decay=1e-4, grad clip=1.0, 1000 epochs
    5 个 inner model 对 fold6 做 ensemble
    RNG: 每个 (D_G, rank) cell 开始前重置 torch.manual_seed(42) / np.random.seed(42)

缓存隔离 (防止冒烟污染正式; control / salt 两个环境各自独立目录):
    outputs/reproduce/dg_rank_rice/ckpt_final/         <- 只有 --run   读写 (--env control)
    outputs/reproduce/dg_rank_rice/ckpt_smoke/         <- 只有 --smoke 读写 (--env control)
    outputs/reproduce/dg_rank_rice_salt/ckpt_final/    <- --env salt 的对应目录
    文件名: dg{DG}_r{rank}_f{fold}_i{inner}_affine_seed42_ep{epochs}.npz
    npz 内 meta 记录 dataset/D_G/rank/fold/inner/seed/epochs/hidden/affine;
    resume 时逐项校验, 与当前配置不一致则拒绝加载。

输出 (outputs/reproduce/dg_rank_rice[_salt]/):
    f{fold}.csv            每个 (D_G, rank): val/test 的 pcc/mse
    agg.csv                完整二维表: val_pcc_mean/std, val_mse_mean/std,
                           test_pcc_mean/std, test_mse_mean/std, n_foldsets
    matrix_val_pcc.csv     验证 PCC 矩阵 (D_G x rank)
    matrix_val_mse.csv     验证 MSE 矩阵
    matrix_test_pcc.csv    测试 PCC 矩阵
    matrix_test_mse.csv    测试 MSE 矩阵
    dataset_manifest.json  数据身份 (文件名/N/T/P/14个DAS/20个性状名)
    run_meta.json          本次运行的协议与网格 (只登记, 不含任何选择结论)

用法:
    python knode_rice_dg_rank_joint.py                   # 只自检(不训练), 默认 control
    python knode_rice_dg_rank_joint.py --run --smoke     # 冒烟: 1 fold x 1 cell x 50 ep, 走 ckpt_smoke
    python knode_rice_dg_rank_joint.py --run             # 全量: 10 fold x 20 cell, 走 ckpt_final
    python knode_rice_dg_rank_joint.py --run --env salt  # 换 salt 环境 (另一套输出目录)
    python knode_rice_dg_rank_joint.py --aggregate-only
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
# ---------------- 数据集身份 (环境相关常量, 由 apply_env() 绑定) ----------------
DATA_SUB = "rice_salinity"                 # 表型 / GRM / CV 所在子目录 (data/ 下)
GRM_FILE = "rice_GRM_226x226.csv"          # 226 x 226, 与 N=226 一一对应

DG_GRID = [10, 20, 30, 40, 50, 80]
RANK_GRID = [1, 2, 3, 4]

# 每个环境一组身份常量; N/T/P 是硬检查值 (已核实 control 与 salt 完全相同)
# DAS_MIN=30: 丢掉 DAS 28, 对齐作者 dynamicGP 的时间窗 (DAS 30-42), 所以 T 由 14 变 13
ENV_CFG = {
    "control": dict(DATASET="rice",
                    PHENO_FILE="rice_indica_control_traits.csv",
                    CV_FILE="rice_cv_10fold.csv",
                    N=226, T=13, P=20, DAS_MIN=30,
                    RUN_SUB="dg_rank_rice"),
    "salt": dict(DATASET="rice_salt",
                 PHENO_FILE="rice_indica_salt_traits.csv",
                 CV_FILE="rice_cv_10fold.csv",          # 与 control 共用同一份折划分
                 N=226, T=13, P=20, DAS_MIN=30,
                 RUN_SUB="dg_rank_rice_salt"),
}

# 下面这些由 apply_env() 赋值 (默认 control)
ENV = "control"
DATASET = None
PHENO_FILE = CV_FILE = None
EXP_N = EXP_T = EXP_P = DAS_MIN = None
RUN_DIR = CKPT_FINAL = CKPT_SMOKE = None


def apply_env(env: str) -> None:
    """按 --env 绑定所有环境相关常量 (数据集名/文件名/硬检查值/输出目录)."""
    global ENV, DATASET, PHENO_FILE, CV_FILE, EXP_N, EXP_T, EXP_P, DAS_MIN, RUN_DIR, CKPT_FINAL, CKPT_SMOKE
    cfg = ENV_CFG[env]
    ENV = env
    DATASET = cfg["DATASET"]
    PHENO_FILE = cfg["PHENO_FILE"]
    CV_FILE = cfg["CV_FILE"]
    EXP_N, EXP_T, EXP_P, DAS_MIN = cfg["N"], cfg["T"], cfg["P"], cfg["DAS_MIN"]
    RUN_DIR = ROOT / "outputs" / "reproduce" / cfg["RUN_SUB"]
    CKPT_FINAL = RUN_DIR / "ckpt_final"
    CKPT_SMOKE = RUN_DIR / "ckpt_smoke"


apply_env("control")


def build_protocol() -> dict:
    """本次运行登记在 run_meta.json 里的协议 (只登记, 不含任何选择结论)."""
    return {
        "dataset": DATASET,
        "environment": ENV,
        "phenotype_file": PHENO_FILE,
        "grm_file": GRM_FILE,
        "cv_file": CV_FILE,
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
    """加载水稻数据; 硬检查 N/T/P 并在第一次加载时写出 dataset_manifest.json."""
    global P, T, N, Y_all, times_all, trait_cols, cv_foldsets, times_t, grm_raw_full
    pheno = pd.read_csv(ROOT / "data" / DATA_SUB / PHENO_FILE)
    pheno = pheno[pheno["DAS"] >= DAS_MIN].copy()   # 丢掉 DAS 28, 对齐作者的 DAS 30-42
    cv_raw = pd.read_csv(ROOT / "data" / DATA_SUB / CV_FILE, index_col=0)
    cv_raw.index = cv_raw.index.astype(str)   # 水稻 CV 的索引列是纯数字 ID, 默认会读成 int64
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


def res_path(fold, smoke: bool) -> Path:
    return RUN_DIR / (f"smoke_f{fold}.csv" if smoke else f"f{fold}.csv")


def run_fold(fold, dgs, ranks, n_epochs, ckpt_dir, smoke):
    csv_path = res_path(fold, smoke)
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    for dg in dgs:
        for rank in ranks:
            rows = [r for r in rows if not (r["D_G"] == dg and r["rank"] == rank)]
            row = run_cell(dg, rank, fold, n_epochs, ckpt_dir)
            rows.append(row)
            pd.DataFrame(rows).sort_values(["D_G", "rank"]).to_csv(csv_path, index=False)
            print(f"  [dg{dg}/r{rank}/f{fold}] val_pcc={row['val_pcc']:.4f} val_mse={row['val_mse']:.6f} "
                  f"| test_pcc={row['test_pcc']:.4f} test_mse={row['test_mse']:.6f}", flush=True)
    return rows


# ---------------- 汇总: 只出表与矩阵, 不做任何选择 ----------------
def aggregate(smoke=False, dgs=None, ranks=None):
    dgs = dgs or DG_GRID
    ranks = ranks or RANK_GRID
    files = sorted(RUN_DIR.glob("smoke_f[0-9].csv" if smoke else "f[0-9].csv"))
    if not files:
        print("没有可汇总的 f*.csv, 先 --run")
        return None
    df = pd.concat([pd.read_csv(f) for f in files])
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
    ap.add_argument("--env", default="control", choices=list(ENV_CFG),
                    help="control (默认, 主图用) | salt; 决定读哪个表型文件与写哪个输出目录")
    a = ap.parse_args()
    apply_env(a.env)          # 必须在用 RUN_DIR / PHENO_FILE / EXP_* 之前调用

    dgs = [int(x) for x in a.dg.split(",")] if a.dg else DG_GRID
    ranks = [int(x) for x in a.ranks.split(",")] if a.ranks else RANK_GRID
    folds = [int(x) for x in a.folds.split(",") if x.strip()]
    epochs = 50 if a.smoke else a.epochs
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

    print(f"Device={DEVICE}  root={ROOT}  env={ENV} ({DATASET})")
    print(f"网格: D_G={dgs} x rank={ranks}  folds={folds}  epochs={epochs}")
    print(f"条件: full  |  drift: affine (含 b_i)  |  seed: {SEED}")
    print(f"缓存: {ckpt_dir}   ({'冒烟专用' if a.smoke else '正式专用'}, 两者不共享)")
    print(f"结果: {RUN_DIR / ('smoke_f{fold}.csv' if a.smoke else 'f{fold}.csv')}")
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
            {**build_protocol(), "grid": dict(D_G=dgs, rank=ranks), "folds": folds, "epochs": epochs},
            indent=2, ensure_ascii=False), encoding="utf-8")

    load_base()
    for fold in folds:
        print(f"\n===== fold-set {fold} =====")
        run_fold(fold, dgs, ranks, epochs, ckpt_dir, a.smoke)
    aggregate(smoke=a.smoke, dgs=dgs, ranks=ranks)


if __name__ == "__main__":
    main()
