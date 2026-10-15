#!/usr/bin/env python3
"""knode_perfect_vs_predicted_operators.py — 玉米: 完美算子 vs 预测算子, 全 rank × 5 cond 的 U/V 遗传力 + 滚验证集.

用户 2026-08-19 设计 (5 步):
  1. 完美算子: 模型在验证集本身 (in_val) 训练 (in-sample), 喂验证集自己的 5 种输入
     (full=grm+tp1, grmonly, tp1only, const, noise) 获得算子, rank ∈ {1,2,4,8,16,32,50},
     算 U/V 遗传力 (按作者方法: 逐 entry HE + GBLUP 5折CV), 存 U/V, 滚验证集 -> MSE/pcc.
     目的: 证明所有 rank 的 U/V 遗传力都很高 (而非作者说的"高 rank 崩掉").
  2. 预测算子: 模型在训练集 (in_tr) 训练 (out-of-sample), 同 rank 集合, 提取 U/V -> 遗传力 ->
     滚验证集 -> MSE/pcc. 这里选最佳 pcc.
  3. dim 扫描: 用步骤2选出的最佳 rank, 对 grmonly 和 full 在 dim∈{10,20,30,40,50} 各跑一遍 -> 最佳 dim.
  4. 图 a: 完美算子所有 rank 的 U/V 遗传力 (都很高) + 滚 pcc (也很高).
     图 b: 预测算子 pcc 明显降低, grm-only 与 const 差距不大 -> 大部分基因型输入影响力不大;
           极高 pcc 说明轨迹可被拟合, 但效果有限, 主要来源是 h0 和模型拟合能力.
  5. 冒烟后先讨论再跑 (--smoke 只跑 1 cond × 1 rank × 少 epochs).

协议: --fold N 选定 CSV 第 N 列 fold-set, 每列固定 in_tr=folds 1:5 (训练) + in_val=fold 6 (验证==测试, 数量逐列不等),
      同作者 id=1:5 循环验证选best + id=6 验证, 全 25 时点训练, seed 42, HID=128, 1000 epochs.
遗传力: 作者方法 GCTA/GREML (单分量 intercept-only REML, 真 GRM) + GBLUP 5折CV (作者 RR-BLUP 对应).
U/V 提取: 喂全 N=330 的输入 -> 遗传力在 N=330 上计算 (作者 Fig.2a 口径), 滚只用验证集.
输出: outputs/reproduce/operator_perfect_predicted_maize_{stage}_f{fold}.csv/.json, uv/*.npz, 图.
"""
import math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from pathlib import Path
import warnings, json, time, argparse; warnings.filterwarnings('ignore')

torch.manual_seed(42); np.random.seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.001; WEIGHT_DECAY = 1e-4
GRM_DIM = 50; HID = 128
RANKS = [1, 2, 4, 8, 16, 32, 50]
CONDS = ['full', 'grmonly', 'tp1only', 'const', 'noise']
FOLD = 0        # 当前 fold-set (CSV 列索引), main 中由 --fold 设置
GRM_DIM_CUR = GRM_DIM   # 当前 grm PCA 实际维度 (dim sweep 时会变)
def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").is_dir() and (p / "outputs").is_dir():
            return p
    raise SystemExit("找不到 knode 根目录 (需要同时存在 data/ 与 outputs/)")


BASE = _find_root(Path(__file__).resolve().parent)   # 位置无关: 不再依赖 CWD
OUT_DIR = BASE / 'outputs' / 'reproduce'; OUT_DIR.mkdir(parents=True, exist_ok=True)
UV_DIR = OUT_DIR / 'uv'; UV_DIR.mkdir(exist_ok=True)
DATA_DIR = BASE / 'data' / 'maize_magic'


# ---------------- data ----------------
def load_data(grm_dim, fold=0):
    global P, T, N, times_t, times_all, trait_cols, grm_raw, grm_pca, noise_all, Y_all, y1_all, cv_foldsets, tr_min, tr_rng
    global in_tr, in_val, FOLD, GRM_DIM_CUR
    FOLD = fold; GRM_DIM_CUR = grm_dim
    pheno = pd.read_csv(DATA_DIR / 'maize_trait_BLUP_50_NArmv_traits_afterClustering.csv')
    cv_raw = pd.read_csv(DATA_DIR / 'cross_validation_folds_w_validation_set_maize.csv', index_col=0)
    grm_full = pd.read_csv(DATA_DIR / 'GRM_330x330_methodA_from552.csv', index_col=0)
    meta_cols = ['bio_ID', 'accession', 'DAS']
    trait_cols = [c for c in pheno.columns if c not in meta_cols and not c.startswith('Unnamed') and c != '']
    P = len(trait_cols)
    times_all = np.array(sorted(pheno['DAS'].unique()), dtype=np.float64)
    T = len(times_all)
    common_ids = sorted(set(pheno['bio_ID'].astype(str)) & set(cv_raw.index))
    N = len(common_ids)
    cv_foldsets = cv_raw.loc[common_ids].iloc[:, :].values.astype(int)
    grm_full.index = grm_full.index.astype(str); grm_full.columns = grm_full.columns.astype(str)
    grm_raw = grm_full.loc[common_ids, common_ids].values.astype(np.float64)
    pca = PCA(n_components=grm_dim)
    grm_pca = pca.fit_transform(grm_raw).astype(np.float32)
    Y_all = np.full((N, T, P), np.nan, dtype=np.float64)
    for pi, trait in enumerate(trait_cols):
        sub = pheno[['bio_ID', 'DAS', trait]].copy(); sub['bio_ID'] = sub['bio_ID'].astype(str)
        Y_all[:, :, pi] = sub.pivot(index='bio_ID', columns='DAS', values=trait).loc[common_ids].values
    assert not np.any(np.isnan(Y_all))
    tn = (times_all - times_all.min()) / (times_all.max() - times_all.min() + 1e-8)
    times_t = torch.tensor(tn, dtype=torch.float32, device=DEVICE)
    # 归一化基准 (train-based, 用该 fold 自己的 in_tr 定义 — 逐 fold 归一化)
    in_tr, in_val = fold_idx(fold)
    tr_min = np.nanmin(Y_all[in_tr], axis=(0, 1)); tr_max = np.nanmax(Y_all[in_tr], axis=(0, 1))
    tr_rng = np.maximum(tr_max - tr_min, 1e-8)
    y1_all = ((Y_all[:, 0, :] - tr_min) / tr_rng).astype(np.float32)
    rng_n = np.random.RandomState(0)
    noise_all = rng_n.randn(N, grm_dim).astype(np.float32)
    return pca.explained_variance_ratio_.sum()


def fold_idx(fs):
    """作者协议: id=1:5 训练, id=6 验证==测试 (验证集就是测试集, 用户明确).
    每列 tr/val 数量不同 (fold6 计数逐列变)."""
    fc = cv_foldsets[:, fs]
    in_tr = np.where((fc >= 1) & (fc <= 5))[0]   # 训练基因型 (folds 1-5)
    in_val = np.where(fc == 6)[0]                 # 验证==测试基因型 (fold 6)
    return in_tr, in_val


# ---------------- hyper ODE (cond_mode 参数化, 含 const/noise) ----------------
class HyperODEModel(nn.Module):
    def __init__(self, P, cond_in, rank, hid=128, cond_mode='full'):
        super().__init__()
        self.P = P; self.rank = rank; self.cond_mode = cond_mode
        self.hyper_enc = nn.Sequential(nn.Linear(cond_in, hid), nn.Tanh())
        self.hyper_U = nn.Linear(hid, P * rank)
        self.hyper_V = nn.Linear(hid, rank * P)
        self._U_dyn = self._V_dyn = None

    def set_z(self, grm_pca, y1):
        if self.cond_mode == 'const':
            B = y1.shape[0] if y1 is not None else grm_pca.shape[0]
            cond = torch.ones(B, GRM_DIM_CUR, device=DEVICE)
        elif self.cond_mode == 'full':
            if grm_pca is not None and not torch.is_tensor(grm_pca):
                grm_pca = torch.tensor(grm_pca, device=y1.device)
            cond = torch.cat([grm_pca, y1], dim=-1)
        elif self.cond_mode in ('grmonly', 'noise'):
            if grm_pca is not None and not torch.is_tensor(grm_pca):
                grm_pca = torch.tensor(grm_pca, device=DEVICE)
            cond = grm_pca
        else:
            cond = y1
        h = self.hyper_enc(cond)
        B = cond.shape[0]
        self._U_dyn = self.hyper_U(h).view(B, self.P, self.rank)
        self._V_dyn = self.hyper_V(h).view(B, self.rank, self.P)

    def forward(self, t, h):
        return torch.bmm(
            torch.bmm(h.unsqueeze(1), self._V_dyn.transpose(1, 2)),
            self._U_dyn.transpose(1, 2)).squeeze(1)


def euler_integrate(model, h0, ts):
    B, D = h0.shape; T_len = len(ts)
    traj = torch.zeros(B, T_len, D, device=h0.device, dtype=h0.dtype); traj[:, 0] = h0
    h = h0
    for i in range(1, T_len):
        dt = max(ts[i].item() - ts[i - 1].item(), 1e-6)
        h = h + model(None, h) * dt; traj[:, i] = h
    return traj


def train_hyper(cond, rank, Ys_tr, g_tr, Ys_early, g_early, seed, n_epochs):
    """全训练 (loss 覆盖 t1..T-1), 内折早停 (Ys_early/g_early 上的 loss).
    g_*: full/grmonly/noise 传 grm/noise 张量, tp1/const 传 None (h0 已有)."""
    torch.manual_seed(seed); np.random.seed(seed)
    B, Tn, Pn = Ys_tr.shape
    if cond == 'full':
        cond_in = GRM_DIM_CUR + Pn
    elif cond in ('grmonly', 'noise', 'const'):
        cond_in = GRM_DIM_CUR
    else:
        cond_in = Pn
    m = HyperODEModel(Pn, cond_in, rank, HID, cond).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best = None; best_loss = float('inf')
    y_tr = torch.tensor(Ys_tr.astype(np.float32), device=DEVICE)
    for ep in range(n_epochs):
        m.train(); m.set_z(g_tr, y_tr[:, 0, :])
        pred = euler_integrate(m, y_tr[:, 0, :], times_t)
        loss = F.mse_loss(pred[:, 1:, :], y_tr[:, 1:, :])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 50 == 0:
            with torch.no_grad():
                m.eval()
                m.set_z(g_early, torch.tensor(Ys_early[:, 0, :].astype(np.float32), device=DEVICE))
                pv = euler_integrate(m, torch.tensor(Ys_early[:, 0, :].astype(np.float32), device=DEVICE), times_t)
                vl = F.mse_loss(pv[:, 1:, :],
                                torch.tensor(Ys_early[:, 1:, :].astype(np.float32), device=DEVICE)).item()
            if not math.isnan(vl) and vl < best_loss:
                best_loss = vl
                best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    if best is None:
        best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best); m.eval()
    return m


def extract_uv_all(model, cond):
    """喂全 N=330 的输入, 提取 U (N,P,r) 和 V (N,r,P)."""
    with torch.no_grad():
        if cond == 'full':
            model.set_z(torch.tensor(grm_pca, device=DEVICE), torch.tensor(y1_all, device=DEVICE))
        elif cond in ('grmonly', 'noise'):
            model.set_z(torch.tensor(grm_pca if cond == 'grmonly' else noise_all, device=DEVICE), None)
        else:  # tp1only / const: const 用 y1 只取 B
            model.set_z(None, torch.tensor(y1_all, device=DEVICE))
        U = model._U_dyn.cpu().numpy().copy()
        V = model._V_dyn.cpu().numpy().copy()
    return U, V


def roll_on_ids(model, cond, ids):
    """h0 = y1[ids] (真), 递归积分全时点. 返回 (n, T, P) 预测轨迹."""
    with torch.no_grad():
        model.eval()
        if cond == 'full':
            model.set_z(torch.tensor(grm_pca[ids], device=DEVICE), torch.tensor(y1_all[ids], device=DEVICE))
        elif cond in ('grmonly', 'noise'):
            model.set_z(torch.tensor((grm_pca if cond == 'grmonly' else noise_all)[ids], device=DEVICE), None)
        else:
            model.set_z(None, torch.tensor(y1_all[ids], device=DEVICE))
        pred = euler_integrate(model, torch.tensor(y1_all[ids], device=DEVICE), times_t).cpu().numpy()
    return pred


def roll_metrics(pred, Y_true):
    """逐时点 pcc (t>=1, 跨 trait 平均) + 全时点 MSE (归一化单位)."""
    n, Tn, Pn = Y_true.shape
    pccs = []
    for t in range(1, Tn):
        rs = []
        for p in range(Pn):
            if np.std(Y_true[:, t, p]) < 1e-12: continue
            rs.append(pearsonr(pred[:, t, p], Y_true[:, t, p])[0])
        pccs.append(float(np.mean(rs)) if rs else float('nan'))
    mse = float(np.mean((pred[:, 1:, :] - Y_true[:, 1:, :]) ** 2))
    return float(np.nanmean(pccs)), pccs, mse


# ---------------- 遗传力 (作者方法 GCTA/GREML: 逐 entry 视为单性状) ----------------
def greml_h2_batch(Ymat, K, ngrid=120):
    """单分量 GREML (intercept-only REML, 同 GCTA --reml 默认口径) 批量: Ymat (N,E) -> (E,) h2.
    模型 y ~ N(1*mu, sig_g2*K + sig_e2*I). 特征分解 K=U diag(w) U^T, 对 r=sig_e2/sig_g2 网格
    求 REML profile (对 sig_g2 有闭式解 sig_g2 = y'Py/(N-1)), 逐列 argmax."""
    N_, E = Ymat.shape
    w, U = np.linalg.eigh(K)              # K = U diag(w) U^T
    w = np.clip(w, 1e-8, None)
    Yt = U.T @ Ymat                       # (N,E) 特征空间投影
    s1 = U.sum(0)                         # U^T 1 (均值方向在特征空间)
    rs = np.geomspace(1e-6, 1e6, ngrid)   # r = sig_e2/sig_g2 网格
    best_h2 = np.zeros(E); best_logL = np.full(E, -np.inf)
    for r in rs:
        d = w + r
        logW = np.sum(np.log(d))
        t1 = np.sum(s1 ** 2 / d)                        # X'W^-1X
        yt2 = np.sum(Yt ** 2 / d[:, None], axis=0)      # y'W^-1y
        B = np.sum(Yt * (s1 / d)[:, None], axis=0)      # y'W^-1X
        yPy = yt2 - B ** 2 / t1                         # 投影二次型 (>=0)
        sg2 = np.maximum(yPy / (N_ - 1), 1e-12)
        logL = -0.5 * ((N_ - 1) * np.log(sg2) + logW + np.log(t1) + (N_ - 1))
        m = logL > best_logL
        best_logL[m] = logL[m]
        best_h2[m] = 1.0 / (1.0 + r)      # h2 = sig_g2/(sig_g2+sig_e2)
    # 常数性状 (逐基因型零方差, 如 const 条件的 U/V) 遗传力无定义: 中心化后 std≈0,
    # profile 在 r 网格上不可辨识, 会被钳制/边界效应推到 h2≈1. 按惯例判 0.
    const_mask = np.nanmax(np.abs(Ymat - Ymat.mean(0)), axis=0) < 1e-12
    best_h2[const_mask] = 0.0
    return np.clip(best_h2, 0.0, 1.0)     # 截到 [0,1] 同 GCTA 输出口径


def gblup_cv_centered_batch(Ymat, K, seed=0, lam=1.0):
    """RR-BLUP/GBLUP 5折CV 精度: 先整体中心化 (同 per-entry), 再批量解. 返回 (E,) Pearson."""
    N_, E = Ymat.shape
    yc = Ymat - Ymat.mean(0)
    kf = KFold(5, shuffle=True, random_state=seed)
    pred = np.zeros_like(yc)
    for tr, te in kf.split(np.arange(N_)):
        Ktt = K[tr][:, tr]; Kte = K[te][:, tr]
        coef = np.linalg.solve(Ktt + lam * np.eye(len(tr)), yc[tr])   # (|tr|, E)
        pred[te] = Kte @ coef
    out = np.full(E, np.nan)
    for j in range(E):
        if np.std(yc[:, j]) < 1e-12 or np.std(pred[:, j]) < 1e-12: continue
        out[j] = pearsonr(yc[:, j], pred[:, j])[0]
    return out


def entry_stats(U, V, grm):
    """U (N,P,r), V (N,r,P): 每 entry 一列做 GREML 遗传力 + GBLUP CV, 返回均值±std."""
    def stat(M):
        E = M.reshape(N, -1)
        h2 = greml_h2_batch(E, grm)
        acc = gblup_cv_centered_batch(E, grm)
        return (float(np.nanmean(h2)), float(np.nanstd(h2)),
                float(np.nanmean(acc)), float(np.nanstd(acc)))
    hU, sU, aU, saU = stat(U)
    hV, sV, aV, saV = stat(V)
    return {'U_h2': round(hU, 4), 'U_h2_std': round(sU, 4), 'U_acc': round(aU, 4), 'U_acc_std': round(saU, 4),
            'V_h2': round(hV, 4), 'V_h2_std': round(sV, 4), 'V_acc': round(aV, 4), 'V_acc_std': round(saV, 4)}


def subset_h2(U, V, grm, idx, tag):
    """U/V 在 idx 基因型子集上的 GREML 遗传力 (GRM 取子矩阵).
    用于比较换 CV 后 U/V 遗传力是否暴跌: 训练基因型 vs 验证/测试基因型."""
    Kr = grm[np.ix_(idx, idx)]
    Ur = U[idx].reshape(len(idx), -1)
    Vr = V[idx].reshape(len(idx), -1)
    def h2mean(M):
        return round(float(np.nanmean(greml_h2_batch(M, Kr))), 4)
    return {f'U_h2_{tag}': h2mean(Ur), f'V_h2_{tag}': h2mean(Vr)}


# ---------------- 阶段 1 & 2: 完美算子 / 预测算子 (共享主循环) ----------------
def run_one(cond, rank, train_ids, early_ids, tag, n_epochs, save_uv):
    """train_ids: 训练集 (完美=in_val, 预测=in_tr); early_ids: 早停集 (完美=in_val, 预测=in_val).
    提取全 N U/V -> 遗传力; 滚验证集 (in_val) -> MSE/pcc. 返回一行结果."""
    Ys_tr = ((Y_all[train_ids] - tr_min) / tr_rng).astype(np.float32)
    Ys_early = ((Y_all[early_ids] - tr_min) / tr_rng).astype(np.float32)
    Ys_val = ((Y_all[in_val] - tr_min) / tr_rng).astype(np.float32)

    if cond == 'full':
        g_tr = torch.tensor(grm_pca[train_ids], device=DEVICE); g_early = torch.tensor(grm_pca[early_ids], device=DEVICE)
    elif cond in ('grmonly', 'noise'):
        src = grm_pca if cond == 'grmonly' else noise_all
        g_tr = torch.tensor(src[train_ids], device=DEVICE); g_early = torch.tensor(src[early_ids], device=DEVICE)
    else:
        g_tr = g_early = None

    m = train_hyper(cond, rank, Ys_tr, g_tr, Ys_early, g_early, 42, n_epochs)

    U, V = extract_uv_all(m, cond)                       # 全 N
    st = entry_stats(U, V, grm_raw)                      # 全 N 遗传力 + GBLUP CV (U/V)
    sub = {}
    sub.update(subset_h2(U, V, grm_raw, in_tr, 'tr'))    # 训练基因型 (训练时遗传力)
    sub.update(subset_h2(U, V, grm_raw, in_val, 'val'))  # 验证==测试基因型
    pred_val = roll_on_ids(m, cond, in_val)              # 滚验证集 (== 测试集, id=6)
    pcc_mean, pcc_per_tp, mse = roll_metrics(pred_val, Ys_val)

    if save_uv:
        np.savez_compressed(UV_DIR / f'{tag}_{cond}_r{rank}.npz',
                            U=U.astype(np.float32), V=V.astype(np.float32))

    return {**{'tag': tag, 'cond': cond, 'rank': rank}, **st, **sub,
            'pcc_val': round(pcc_mean, 4),
            'pcc_per_tp': [round(x, 4) for x in pcc_per_tp],
            'mse_val': round(mse, 6)}


def stage_perfect_predicted(tag, train_ids, early_ids, conds, ranks, n_epochs, save_uv):
    print(f'\n=== {tag}: train_ids={train_ids} early_ids={early_ids} ===', flush=True)
    rows = []
    for cond in conds:
        for rank in ranks:
            print(f'\n  [{tag}] cond={cond} rank={rank}', flush=True)
            t0 = time.time()
            r = run_one(cond, rank, train_ids, early_ids, tag, n_epochs, save_uv)
            rows.append(r)
            print(f'    U_h2={r["U_h2"]} V_h2={r["V_h2"]} U_acc={r["U_acc"]} V_acc={r["V_acc"]} '
                  f'| 六点: Utr={r["U_h2_tr"]} Uval={r["U_h2_val"]} Vtr={r["V_h2_tr"]} Vval={r["V_h2_val"]} '
                  f'| pcc_val={r["pcc_val"]} mse={r["mse_val"]} ({time.time()-t0:.1f}s)',
                  flush=True)
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f'operator_{tag}_maize_f{FOLD}.csv', index=False)
    with open(OUT_DIR / f'operator_{tag}_maize_f{FOLD}.json', 'w') as f:
        json.dump({'rows': rows}, f, indent=2)
    return df


# ---------------- 阶段 3: dim 扫描 (最佳 rank, grmonly + full) ----------------
def stage_dim_sweep(best_rank, dims, n_epochs):
    global grm_pca, noise_all
    rows = []
    for cond in ['grmonly', 'full']:
        for dim in dims:
            print(f'\n  [DimSweep] cond={cond} dim={dim} rank={best_rank}', flush=True)
            ev = load_data(dim, FOLD)    # 重新 PCA 到 dim (重载数据, grm_pca/noise_all 更新), 保持同 fold
            t0 = time.time()
            r = run_one(cond, best_rank, in_tr, in_val, f'dim{cond}', n_epochs, save_uv=False)
            r = {**{'cond': cond, 'grm_dim': dim, 'pca_ev': round(ev, 3)}, **{k: v for k, v in r.items() if k not in ('tag', 'cond', 'rank')}}
            r['rank'] = best_rank
            rows.append(r)
            print(f'    pcc_val={r["pcc_val"]} U_h2={r["U_h2"]} ({time.time()-t0:.1f}s)', flush=True)
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f'operator_dimsweep_maize_f{FOLD}.csv', index=False)
    with open(OUT_DIR / f'operator_dimsweep_maize_f{FOLD}.json', 'w') as f:
        json.dump({'rows': rows}, f, indent=2)
    return df


# ---------------- 阶段 4: 图 ----------------
def plot_figs():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p1 = pd.read_csv(OUT_DIR / f'operator_perfect_maize_f{FOLD}.csv')
    p2 = pd.read_csv(OUT_DIR / f'operator_predicted_maize_f{FOLD}.csv')

    # Fig a: 完美算子 — U/V 遗传力 + 滚 pcc 全 rank
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for cond in CONDS:
        d = p1[p1.cond == cond]
        ax.plot(d['rank'], d['U_h2'], marker='o', label=f'{cond} U h2')
    ax.set_xscale('log', base=2); ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
    ax.set_xlabel('rank'); ax.set_ylabel('entry heritability (GREML)')
    ax.set_title('Perfect operators: U heritability vs rank'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax = axes[1]
    for cond in CONDS:
        d = p1[p1.cond == cond]
        ax.plot(d['rank'], d['pcc_val'], marker='o', label=cond)
    ax.set_xscale('log', base=2); ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
    ax.set_xlabel('rank'); ax.set_ylabel('roll pcc (val)')
    ax.set_title('Perfect operators: roll pcc vs rank'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / f'fig_operator_perfect_a_f{FOLD}.png', dpi=150); plt.close()

    # Fig b: 预测算子 — 滚 pcc 每 cond (完美算子作参考)
    conds = [c for c in CONDS if c in p2.cond.values]
    best1 = p1.sort_values('pcc_val', ascending=False).drop_duplicates('cond')   # 每 cond 最佳 rank
    best2 = p2.sort_values('pcc_val', ascending=False).drop_duplicates('cond')
    def get(df, c, col='pcc_val'):
        return float(df.loc[df.cond == c, col].iloc[0])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(conds))
    ax[0].bar(x - 0.2, [get(best2, c) for c in conds], 0.4, label='predicted (best rank)', alpha=.8)
    ax[0].bar(x + 0.2, [get(best1, c) for c in conds], 0.4, label='perfect (best rank)', alpha=.6)
    ax[0].set_xticks(x); ax[0].set_xticklabels(conds, rotation=20)
    ax[0].set_ylabel('roll pcc (val)'); ax[0].set_title('Predicted vs perfect operator pcc')
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    # grm-only vs const 差距 (预测算子): 遗传信息带来的提升相当有限
    if 'grmonly' in conds and 'const' in conds:
        ax[1].plot([0, 1], [get(best2, 'grmonly'), get(best2, 'const')], 'o-', label='grmonly vs const')
        ax[1].annotate(f'grmonly {get(best2, "grmonly"):.3f}', (0.0, get(best2, 'grmonly')))
        ax[1].annotate(f'const {get(best2, "const"):.3f}', (0.55, get(best2, 'const')))
    ax[1].set_xticks([]); ax[1].set_ylabel('roll pcc (val)')
    ax[1].set_title('grm 带来的遗传信息相当有限 (预测算子)'); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / f'fig_operator_predicted_b_f{FOLD}.png', dpi=150); plt.close()
    print(f'  figs -> fig_operator_perfect_a_f{FOLD}.png / fig_operator_predicted_b_f{FOLD}.png', flush=True)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='冒烟: 1 cond × 1 rank × 5 epochs')
    ap.add_argument('--stages', default='1234', help='运行的阶段 (1=完美,2=预测,3=dim,4=图)')
    ap.add_argument('--n_epochs', type=int, default=1000)
    ap.add_argument('--dims', type=str, default='10,20,30,40,50')
    ap.add_argument('--fold', type=int, default=0, help='fold-set 列索引 (CSV 第 N 列, 0-9)')
    args = ap.parse_args()

    conds = ['full'] if args.smoke else CONDS
    ranks = [2] if args.smoke else RANKS
    n_epochs = 5 if args.smoke else args.n_epochs
    dims = [10] if args.smoke else [int(x) for x in args.dims.split(',')]

    ev = load_data(GRM_DIM, args.fold)
    in_tr, in_val = fold_idx(args.fold)
    print(f'Device: {DEVICE}  fold={args.fold}  N={N} P={P} T={T} GRM_PCA_var(50)={ev:.3f}  '
          f'in_tr={len(in_tr)} in_val==test={len(in_val)}', flush=True)

    if '1' in args.stages:
        df1 = stage_perfect_predicted('perfect', in_val, in_val, conds, ranks, n_epochs, save_uv=True)
    if '2' in args.stages:
        df2 = stage_perfect_predicted('predicted', in_tr, in_val, conds, ranks, n_epochs, save_uv=True)
        best = df2.loc[df2.pcc_val.idxmax()]
        print(f'\n>> 最佳 pcc (预测算子): cond={best["cond"]} rank={best["rank"]} pcc_val={best["pcc_val"]}', flush=True)
    if '3' in args.stages and not args.smoke:
        best_rank = int(best['rank'])
        stage_dim_sweep(best_rank, dims, n_epochs)
    if '4' in args.stages:
        try:
            plot_figs()
        except Exception as e:
            print(f'  [warn] fig stage failed: {e}', flush=True)
    print('\nDone.')


if __name__ == '__main__':
    main()
