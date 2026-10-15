# -*- coding: utf-8 -*-
"""knode_rank_genetics_value_arab.py -- 拟南芥 Fig.4 的 rank 扫描数据 (rank sweep, 十折).

每个 (cond, rank) 训一个 hyper-ODE, 抽 U/V 算子, 算三件事:
  1. heritability  : GREML h2 of U/V / A 的元
  2. predictability: GBLUP 5 折 (within) + GBLUP train->test 外推 (ext_A 供 Fig.4 panel b)
  3. value         : end-to-end rollout PCC / MSE (供 Fig.4 panel a)
conditions: full / grmonly / tp1only / const / noise

锁定配置 (2026-09-19): dataset=arab, D_G=30, rank 扫描 1..45 (=P), HID=128, Tanh,
    affine drift b_i (只含 b_i 这一项加性漂移), 归一化真实时间间隔 explicit Euler, 整轨迹 MSE,
    checkpoint selection: 每 50 epochs 取验证集(剔 t1)MSE 最低的权重, 跑满 1000 epochs (无提前终止),
    inner-5 集成 (fold1-5 训练, fold6 测试), 10 fold-set, seed=42。

不覆盖 / 不串味规则 (与 cond ladder / dg_rank 一致):
    RUN_DIR = outputs/reproduce/rank_sweep_arab_dg30/
        cells/f{fold}_{cond}_r{rank}_ep{epochs}.csv   每个 cell 一个文件; **已存在就跳过, 绝不重算/覆盖**
        cache/f{fold}_{cond}_r{rank}_i{inner}_ep{epochs}.npz   per-inner 缓存, 内含 meta, 不一致拒绝加载
        agg.csv / dataset_manifest.json / run_meta.json
        smoke_cells/ + cache_smoke/   --smoke 专用, 永远不会被 aggregate 读到
    **绝对不读旧的 outputs/reproduce/uv2_arab/ (旧口径缓存, 无 meta, 已作废)**
    想重算某 cell: 删掉它那个 cells csv (和对应 cache npz) 再跑。

用法 (CWD 不限):
    python knode_rank_genetics_value_arab.py                 # 全量: 10 fold x 5 cond x 11 rank
    python knode_rank_genetics_value_arab.py --smoke         # 冒烟: 1 fold x 1 cond x 1 rank x 50 ep
    python knode_rank_genetics_value_arab.py --folds 0 --ranks 1,2 --conds full
    python knode_rank_genetics_value_arab.py --aggregate
"""
import math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from pathlib import Path
import argparse, warnings, time, json
warnings.filterwarnings("ignore")
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # reproduce/ 在上一级: knode_perfect_vs_predicted_operators 在那里
from knode_perfect_vs_predicted_operators import greml_h2_batch, gblup_cv_centered_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3; WD = 1e-4; HID = 128
DG_DEFAULT = 30                # 锁定 D_G (2026-09-19 由 D_G x rank 联合扫描人工判断)
GRM_DIM = DG_DEFAULT
CONDS = ["full", "grmonly", "tp1only", "const", "noise"]
RANKS = [1, 2, 3, 4, 6, 8, 10, 16, 20, 30, 45]     # 一直到 P=45
N_EPOCHS_DEFAULT = 1000        # 正式结果只允许这个值 (短跑请用 --smoke)
SEED = 42
DATASET = "arab"
EXP_N, EXP_T, EXP_P = 382, 14, 45                  # 数据身份硬检查
SMOKE_MODE = False


def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").is_dir() and (p / "outputs").is_dir():
            return p
    raise SystemExit("找不到 knode 根目录 (需要同时存在 data/ 与 outputs/)")


BASE = _find_root(Path(__file__).resolve().parent)
ENV = "drought"          # 只有 UAV 用 (drought|irrigated); arab/rice 固定 drought


def run_dir() -> Path:
    """每个 (数据集, 环境, D_G) 一个独立 run directory: 不同配置永不互相覆盖。"""
    sub = f"rank_sweep_{DATASET}_dg{GRM_DIM}" + ("" if ENV == "drought" else "_" + ENV)
    d = BASE / "outputs" / "reproduce" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def cell_dir() -> Path:
    d = run_dir() / ("smoke_cells" if SMOKE_MODE else "cells")
    d.mkdir(parents=True, exist_ok=True); return d


def cache_dir() -> Path:
    d = run_dir() / ("cache_smoke" if SMOKE_MODE else "cache")
    d.mkdir(parents=True, exist_ok=True); return d


def cell_csv(fold, cond, rank, epochs) -> Path:
    return cell_dir() / f"f{fold}_{cond}_r{rank}_ep{epochs}.csv"


def ck_path(fold, cond, rank, inner, epochs) -> Path:
    return cache_dir() / f"f{fold}_{cond}_r{rank}_i{inner}_ep{epochs}.npz"


def cell_meta(fold, cond, rank, inner, epochs) -> dict:
    return dict(dataset=DATASET, D_G=int(GRM_DIM), rank=int(rank), cond=str(cond),
                fold=int(fold), inner=int(inner), seed=SEED, epochs=int(epochs),
                hidden=HID, drift="affine",
                # 无个体信息档的输入常数 (2026-09-20 起统一为 0.0; 改定义会导致旧 cache 拒绝加载, 必须换目录重跑)
                noinfo_input=(0.0 if cond == "const" else None))


def load_ckpt(path: Path, want: dict):
    """resume: meta 逐项一致才允许读; 旧格式(无 meta)一律拒绝。"""
    z = np.load(path, allow_pickle=True)
    if "meta" not in z:
        raise RuntimeError(f"{path.name} 无 meta (旧缓存), 拒绝加载")
    got = json.loads(str(z["meta"][0]))
    diff = {k: (got.get(k), want[k]) for k in want if got.get(k) != want[k]}
    if diff:
        raise RuntimeError(f"{path.name} meta 与当前配置不一致 {diff}, 拒绝加载")
    return z["U"], z["V"], z["pred"]


_MANIFEST_WRITTEN = False


def check_data():
    """硬检查 N/T/P + 写 dataset_manifest.json (只写一次)。"""
    global _MANIFEST_WRITTEN
    assert P == EXP_P, f"Unexpected trait number: P={P}, expected {EXP_P}"
    assert T == EXP_T, f"Unexpected time points: T={T}, expected {EXP_T}"
    assert N == EXP_N, f"Unexpected genotype number: N={N}, expected {EXP_N}"
    if _MANIFEST_WRITTEN:
        return
    man = dict(dataset=DATASET, N=int(N), T=int(T), P=int(P), D_G=int(GRM_DIM),
               DAS=[float(x) for x in times_all], trait_names=list(trait_cols),
               cv_foldsets=int(cv_foldsets.shape[1]))
    (run_dir() / "dataset_manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    _MANIFEST_WRITTEN = True
    print(f"[data] 硬检查通过: N={N} T={T} P={P} | D_G={GRM_DIM} -> dataset_manifest.json", flush=True)

torch.manual_seed(42); np.random.seed(42)

P = T = N = None
grm_pca = y1_all = noise_all = Y_all = times_t = grm_raw = cv_foldsets = tr_min = tr_rng = None
noise2_all = None


def load_data(grm_dim, fold=0):
    global P, T, N, times_t, times_all, trait_cols, grm_raw, grm_pca, noise_all, Y_all, y1_all, cv_foldsets, tr_min, tr_rng
    pheno = pd.read_csv(BASE / "data" / "arabidopsis" / "BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv")
    cv_raw = pd.read_csv(BASE / "data" / "arabidopsis" / "cross_validation_folds_w_validation_set_arabidopsis.csv", index_col=0)
    grm_full = pd.read_csv(BASE / "data" / "arabidopsis" / "arabidopsis_GRM_382.csv", index_col=0)
    meta_cols = ["bio_ID", "accession", "DAS"]
    trait_cols = [c for c in pheno.columns if c not in meta_cols and not c.startswith("Unnamed") and c != ""]
    P = len(trait_cols)
    times_all = np.array(sorted(pheno["DAS"].unique()), dtype=np.float64)
    T = len(times_all)
    common_ids = sorted(set(pheno["bio_ID"].astype(str)) & set(cv_raw.index))
    N = len(common_ids)
    cv_foldsets = cv_raw.loc[common_ids].iloc[:, :].values.astype(int)
    grm_full.index = grm_full.index.astype(str); grm_full.columns = grm_full.columns.astype(str)
    grm_raw = grm_full.loc[common_ids, common_ids].values.astype(np.float64)
    pca = PCA(n_components=grm_dim)
    grm_pca = pca.fit_transform(grm_raw).astype(np.float32)
    Y_all = np.full((N, T, P), np.nan, dtype=np.float64)
    for pi, trait in enumerate(trait_cols):
        sub = pheno[["bio_ID", "DAS", trait]].copy(); sub["bio_ID"] = sub["bio_ID"].astype(str)
        Y_all[:, :, pi] = sub.pivot(index="bio_ID", columns="DAS", values=trait).loc[common_ids].values
    assert not np.any(np.isnan(Y_all))
    tn = (times_all - times_all.min()) / (times_all.max() - times_all.min() + 1e-8)
    times_t = torch.tensor(tn, dtype=torch.float32, device=DEVICE)
    in_tr, in_val = fold_idx(fold)
    tr_min = np.nanmin(Y_all[in_tr], axis=(0, 1)); tr_max = np.nanmax(Y_all[in_tr], axis=(0, 1))
    tr_rng = np.maximum(tr_max - tr_min, 1e-8)
    y1_all = ((Y_all[:, 0, :] - tr_min) / tr_rng).astype(np.float32)
    noise_all = np.random.RandomState(0).randn(N, grm_dim).astype(np.float32)
    return pca.explained_variance_ratio_.sum()


def fold_idx(fs):
    fc = cv_foldsets[:, fs]
    in_tr = np.where((fc >= 1) & (fc <= 5))[0]
    in_val = np.where(fc == 6)[0]
    return in_tr, in_val


class MaskedHyperODEModel(nn.Module):
    def __init__(self, P, rank, hid=128):
        super().__init__()
        self.P = P; self.rank = rank
        self.cond_in = GRM_DIM + P
        self.hyper_enc = nn.Sequential(nn.Linear(self.cond_in, hid), nn.Tanh())
        self.hyper_U = nn.Linear(hid, P * rank)
        self.hyper_V = nn.Linear(hid, rank * P)
        self.hyper_b = nn.Linear(hid, P)          # genotype-specific affine drift b_i
        self._U_dyn = self._V_dyn = self._b_dyn = None

    def set_z(self, grm_chan, t1_chan):
        if grm_chan is not None and not torch.is_tensor(grm_chan):
            grm_chan = torch.tensor(grm_chan, device=DEVICE)
        if t1_chan is not None and not torch.is_tensor(t1_chan):
            t1_chan = torch.tensor(t1_chan, device=DEVICE)
        B = grm_chan.shape[0] if grm_chan is not None else t1_chan.shape[0]
        g = grm_chan if grm_chan is not None else torch.zeros(B, GRM_DIM, device=DEVICE)
        t = t1_chan if t1_chan is not None else torch.zeros(B, self.P, device=DEVICE)
        cond = torch.cat([g, t], dim=-1)
        h = self.hyper_enc(cond)
        self._U_dyn = self.hyper_U(h).view(B, self.P, self.rank)
        self._V_dyn = self.hyper_V(h).view(B, self.rank, self.P)
        self._b_dyn = self.hyper_b(h).view(B, self.P)

    def forward(self, t, h):
        lin = torch.bmm(torch.bmm(h.unsqueeze(1), self._V_dyn.transpose(1, 2)),
                        self._U_dyn.transpose(1, 2)).squeeze(1)
        return lin + self._b_dyn


def euler_integrate(model, h0, ts):
    B, D = h0.shape; T_len = len(ts)
    traj = torch.zeros(B, T_len, D, device=h0.device, dtype=h0.dtype); traj[:, 0] = h0
    h = h0
    for i in range(1, T_len):
        dt = max(ts[i].item() - ts[i - 1].item(), 1e-6)
        h = h + model(None, h) * dt
        traj[:, i] = h
    return traj


def channels(cond, ids):
    ids = np.asarray(ids)
    if cond == "full":
        g = torch.tensor(grm_pca[ids], device=DEVICE); t = torch.tensor(y1_all[ids], device=DEVICE)
    elif cond == "grmonly":
        g = torch.tensor(grm_pca[ids], device=DEVICE); t = None
    elif cond == "tp1only":
        g = None; t = torch.tensor(y1_all[ids], device=DEVICE)
    elif cond == "const":
        # 2026-09-20: 与图三的 shared 统一 —— 无个体信息输入用**全 0** (原来写的是全 1)
        g = torch.zeros(len(ids), GRM_DIM, device=DEVICE); t = torch.zeros(len(ids), P, device=DEVICE)
    elif cond == "noise":
        g = torch.tensor(noise_all[ids], device=DEVICE); t = torch.tensor(noise2_all[ids], device=DEVICE)
    else:
        raise ValueError(cond)
    return g, t


def train_model(cond, rank, Ys_tr, train_ids, Ys_early, early_ids, n_epochs):
    m = MaskedHyperODEModel(P, rank, HID).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    y_tr = torch.tensor(Ys_tr.astype(np.float32), device=DEVICE)
    g_tr, t_tr = channels(cond, train_ids)
    y_early = torch.tensor(Ys_early.astype(np.float32), device=DEVICE)
    g_early, t_early = channels(cond, early_ids)
    best = None; best_loss = float("inf")
    for ep in range(n_epochs):
        m.train(); m.set_z(g_tr, t_tr)
        pred = euler_integrate(m, y_tr[:, 0, :], times_t)
        loss = F.mse_loss(pred, y_tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 50 == 0 or ep == n_epochs - 1:
            with torch.no_grad():
                m.eval(); m.set_z(g_early, t_early)
                pv = euler_integrate(m, y_early[:, 0, :], times_t)
                vl = F.mse_loss(pv[:, 1:, :], y_early[:, 1:, :]).item()
            if best is None or (not math.isnan(vl) and vl < best_loss):
                best_loss = vl; best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best); m.eval()
    return m


def extract_uv(m, cond):
    ids = np.arange(N)
    g, t = channels(cond, ids)
    with torch.no_grad():
        m.set_z(g, t)
        U = m._U_dyn.cpu().numpy().copy()
        V = m._V_dyn.cpu().numpy().copy()
    return U, V


def predict_torch(m, cond, ids):
    """Predict with the model itself (set_z -> explicit Euler rollout from the normalized t1).
    This is the only path that keeps the genotype-specific affine drift b_i."""
    g, t = channels(cond, ids)
    with torch.no_grad():
        m.eval(); m.set_z(g, t)
        y1 = torch.tensor(y1_all[ids], device=DEVICE)
        out = euler_integrate(m, y1, times_t)
    return out.cpu().numpy()


def roll_pred(K, ids):
    """Numpy rollout of the extracted operator K = U V (explicit Euler, same dt as the main script)."""
    y1 = y1_all[ids]
    pred = np.zeros((len(ids), T, P), dtype=np.float64); pred[:, 0, :] = y1
    h = y1
    tn = times_t.cpu().numpy()
    for i in range(1, T):
        dt = max(tn[i] - tn[i - 1], 1e-6)
        h = h + np.einsum('bp,bqp->bq', h, K[ids]) * dt
        pred[:, i, :] = h
    return pred


def pcc_mse_from_pred(pred, Y_true):
    """Snapshot PCC / MSE of a (already ensembled) trajectory, t0 excluded."""
    pccs = []
    for t in range(1, T):
        rs = []
        for p in range(P):
            if np.std(Y_true[:, t, p]) < 1e-12: continue
            rs.append(pearsonr(pred[:, t, p], Y_true[:, t, p])[0])
        pccs.append(float(np.mean(rs)) if rs else float('nan'))
    mse = float(np.mean((pred[:, 1:, :] - Y_true[:, 1:, :]) ** 2))
    return float(np.nanmean(pccs)), mse


def gblup_extrap(Umat, in_tr, in_val, grm, lam=1.0):
    E = Umat.shape[1]
    ytr = Umat[in_tr] - Umat[in_tr].mean(0)
    Ktt = grm[np.ix_(in_tr, in_tr)]; Kte = grm[np.ix_(in_val, in_tr)]
    coef = np.linalg.solve(Ktt + lam * np.eye(len(in_tr)), ytr)
    pred = Kte @ coef
    out = np.full(E, np.nan)
    for j in range(E):
        if np.std(ytr[:, j]) < 1e-12 or np.std(Umat[in_val, j]) < 1e-12 or np.std(pred[:, j]) < 1e-12:
            continue
        out[j] = pearsonr(Umat[in_val, j], pred[:, j])[0]
    return out


def entry_stats(U, V, grm, in_tr, in_val):
    Ur = U.reshape(N, -1).astype(np.float64); Vr = V.reshape(N, -1).astype(np.float64)
    A = np.einsum("npq,nqr->npr", U, V).reshape(N, -1).astype(np.float64)   # A=U·V 逐元素(可识别)
    r = U.shape[2]
    sig = np.zeros((N, r), dtype=np.float64)                                 # A 的奇异值(可识别)
    for i in range(N):
        s = np.linalg.svd(U[i] @ V[i], compute_uv=False)
        sig[i] = s[:r]
    def h2(X): return float(np.nanmean(greml_h2_batch(X, grm)))
    def cv(X): return float(np.nanmean(gblup_cv_centered_batch(X, grm)))
    def ext(X): return float(np.nanmean(gblup_extrap(X, in_tr, in_val, grm)))
    return dict(h2_U=h2(Ur), h2_V=h2(Vr), cv_U=cv(Ur), cv_V=cv(Vr), ext_U=ext(Ur), ext_V=ext(Vr),
                h2_A=h2(A), cv_A=cv(A), ext_A=ext(A),
                h2_sig=h2(sig), cv_sig=cv(sig), ext_sig=ext(sig))


def run_fold(fold, ranks, conds, n_epochs):
    """跑一个 fold-set; 每个 (cond, rank) 一个独立 csv, 已存在就跳过 (绝不覆盖)。"""
    global P, T, N, noise2_all
    load_data(GRM_DIM, fold)
    check_data()
    noise2_all = np.random.RandomState(7).randn(N, P).astype(np.float32)
    in_tr, in_val = fold_idx(fold)
    fc = cv_foldsets[in_tr, fold]                 # 折内标签 1..5 (fold6 已被 fold_idx 排除)
    Ys_val = ((Y_all[in_val] - tr_min) / tr_rng).astype(np.float32)
    rows = []
    for cond in conds:
        for rank in ranks:
            p = cell_csv(fold, cond, rank, n_epochs)
            if p.exists():
                row = pd.read_csv(p).to_dict("records")[0]
                rows.append(row)
                print(f"  [{cond}/r{rank}/f{fold}] 已有结果, 跳过不覆盖 -> {p.name}", flush=True)
                continue
            torch.manual_seed(SEED); np.random.seed(SEED)   # 每个 cell 从同一 RNG 状态开始
            preds, stats = [], []
            for inner in range(1, 6):
                itr = in_tr[fc != inner]; iva = in_tr[fc == inner]
                Ys_in = ((Y_all[itr] - tr_min) / tr_rng).astype(np.float32)
                Ys_ea = ((Y_all[iva] - tr_min) / tr_rng).astype(np.float32)
                f = ck_path(fold, cond, rank, inner, n_epochs)
                if f.exists():
                    U, V, pv = load_ckpt(f, cell_meta(fold, cond, rank, inner, n_epochs))
                else:
                    t0 = time.time()
                    m = train_model(cond, rank, Ys_in, itr, Ys_ea, iva, n_epochs)
                    U, V = extract_uv(m, cond)
                    pv = predict_torch(m, cond, in_val)
                    tmp = f.with_name(f.stem + "__tmp.npz")   # 必须以 .npz 结尾, 否则 numpy 会自动补成 .npz.tmp.npz
                    np.savez_compressed(tmp, U=U.astype(np.float32), V=V.astype(np.float32),
                                        pred=pv.astype(np.float32),
                                        meta=np.array([json.dumps(cell_meta(fold, cond, rank, inner, n_epochs))], dtype=object))
                    tmp.replace(f)                      # 原子落盘
                    print(f'  [train {cond}/r{rank}/inner{inner}] {time.time()-t0:.1f}s', flush=True)
                    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                preds.append(pv)
                stats.append(entry_stats(U, V, grm_raw, in_tr, in_val))
            pcc, mse = pcc_mse_from_pred(np.mean(np.stack(preds), axis=0), Ys_val)
            st = {k: float(np.nanmean([s[k] for s in stats])) for k in stats[0]}
            if cond == 'const':
                st['h2_U'] = 0.0; st['h2_V'] = 0.0; st['h2_A'] = 0.0; st['h2_sig'] = 0.0
            row = {'dataset': DATASET, 'cond': cond, 'rank': rank, 'fold': fold, 'epochs': n_epochs,
                   'grm_dim': GRM_DIM,
                   'h2_U': round(st['h2_U'], 4), 'h2_V': round(st['h2_V'], 4),
                   'cv_U': round(st['cv_U'], 4), 'cv_V': round(st['cv_V'], 4),
                   'ext_U': round(st['ext_U'], 4), 'ext_V': round(st['ext_V'], 4),
                   'h2_A': round(st['h2_A'], 4), 'cv_A': round(st['cv_A'], 4), 'ext_A': round(st['ext_A'], 4),
                   'h2_sig': round(st['h2_sig'], 4), 'cv_sig': round(st['cv_sig'], 4), 'ext_sig': round(st['ext_sig'], 4),
                   'pcc': round(pcc, 4), 'mse': round(mse, 6)}
            tmp = p.with_suffix(".csv.tmp")
            pd.DataFrame([row]).to_csv(tmp, index=False); tmp.replace(p)
            rows.append(row)
            print(f'  [{cond}/r{rank}] h2_U={st["h2_U"]:.3f} ext_U={st["ext_U"]:.3f} pcc={pcc:.4f}', flush=True)
    return rows


def aggregate():
    """只做派生汇总: 读 cells/*.csv, 汇总前先检查口径唯一。"""
    files = sorted(cell_dir().glob("*.csv"))
    if not files:
        print("[warn] cells/ 里没有 cell csv, 先 --run (或先跑 --smoke)"); return None
    gdf = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for col in ("fold", "cond", "rank", "epochs"):
        assert col in gdf.columns, f"cell csv 缺列 {col}"
    eps = sorted(set(int(x) for x in gdf["epochs"]))
    assert len(eps) == 1, f"cells/ 里混了多个 epochs={eps}, 拒绝汇总"
    dup = int(gdf.duplicated(subset=["fold", "cond", "rank"]).sum())
    assert dup == 0, f"有 {dup} 条重复的 (fold, cond, rank), 拒绝汇总"
    gagg = gdf.groupby(["cond", "rank"]).agg(
        h2_U_mean=("h2_U", "mean"), h2_U_std=("h2_U", "std"),
        h2_V_mean=("h2_V", "mean"), h2_V_std=("h2_V", "std"),
        cv_U_mean=("cv_U", "mean"), cv_V_mean=("cv_V", "mean"),
        ext_U_mean=("ext_U", "mean"), ext_U_std=("ext_U", "std"),
        ext_V_mean=("ext_V", "mean"), ext_V_std=("ext_V", "std"),
        h2_A_mean=("h2_A", "mean"), h2_A_std=("h2_A", "std"),
        cv_A_mean=("cv_A", "mean"), ext_A_mean=("ext_A", "mean"), ext_A_std=("ext_A", "std"),
        h2_sig_mean=("h2_sig", "mean"), h2_sig_std=("h2_sig", "std"),
        cv_sig_mean=("cv_sig", "mean"), ext_sig_mean=("ext_sig", "mean"), ext_sig_std=("ext_sig", "std"),
        pcc_mean=("pcc", "mean"), pcc_std=("pcc", "std"),
        mse_mean=("mse", "mean"), mse_std=("mse", "std"), n=("pcc", "count")).reset_index()
    gagg.loc[gagg["cond"] == "const", ["h2_U_mean", "h2_U_std", "h2_V_mean", "h2_V_std", "h2_A_mean", "h2_A_std", "h2_sig_mean", "h2_sig_std"]] = 0.0
    out = run_dir() / ("smoke_agg.csv" if SMOKE_MODE else "agg.csv")
    gagg.to_csv(out, index=False)
    print(f"\n汇总 {len(files)} 个 cell | epochs={eps[0]} | folds={sorted(set(int(x) for x in gdf['fold']))}",
          flush=True)
    print(f"Aggregated -> {out}", flush=True)
    return gagg


def main():
    global SMOKE_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--n_epochs", type=int, default=N_EPOCHS_DEFAULT)
    ap.add_argument("--ranks", type=str, default=None)
    ap.add_argument("--conds", type=str, default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟: 1 fold x 1 cond x 1 rank x 50 epochs, 只写 smoke_cells/ + cache_smoke/")
    args = ap.parse_args()
    ranks = [int(x) for x in args.ranks.split(",")] if args.ranks else RANKS
    conds = args.conds.split(",") if args.conds else CONDS
    epochs = args.n_epochs
    if args.smoke:
        SMOKE_MODE = True
        folds = [int(args.folds.split(",")[0])]
        ranks, conds = ranks[:1], conds[:1]
        epochs = 50 if args.n_epochs == N_EPOCHS_DEFAULT else args.n_epochs
    else:
        if args.n_epochs != N_EPOCHS_DEFAULT:
            raise SystemExit(f"正式结果只允许 n_epochs={N_EPOCHS_DEFAULT} (防短跑污染正式结果); 短跑请加 --smoke")
        folds = [int(x) for x in args.folds.split(",") if x.strip()]
    if args.aggregate:
        aggregate(); return
    print(f"Device={DEVICE} | dataset={DATASET} D_G={GRM_DIM} | folds={folds} | ranks={ranks} | conds={conds} "
          f"| epochs={epochs} | smoke={SMOKE_MODE}", flush=True)
    print(f"run dir = {run_dir()}", flush=True)
    if not SMOKE_MODE:
        (run_dir() / "run_meta.json").write_text(json.dumps(dict(
            dataset=DATASET, D_G=int(GRM_DIM), ranks=list(ranks), conds=list(conds),
            folds=list(folds), epochs=int(epochs), hidden=HID, drift="affine",
            solver="explicit_euler_normalized_real_gaps", loss="mse_full_trajectory",
            checkpoint_selection="best validation MSE over every-50-epoch checks; full 1000 epochs, "
                                 "no patience-based early stopping",
            inner_folds=5, outer_foldsets=10, seed=SEED,
            ensemble="5 inner models averaged on fold6", eval_excludes_t1=True,
            note="只输出结果, 不自动筛参; rank 结论按低秩饱和表述",
            time=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2, ensure_ascii=False), encoding="utf-8")
    for fold in folds:
        print(f"\n===== fold {fold} =====", flush=True)
        run_fold(fold, ranks, conds, epochs)
    aggregate()


if __name__ == "__main__":
    main()
