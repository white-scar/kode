# -*- coding: utf-8 -*-
"""knode_extrap_maize_arab_v2.py - Kode 外推 (rank2 + skip + affine), 十折, 拟南芥 dim30.

模型结构约束 (全场景一致):
  超网络输入 = [grm_pca, x_t1]  ->  输出算子 (U,V,b), 训练完冻结;
  任何场景都不会把 tpN 喂进超网络.
  算子一步: x_{t+1} = A(x_t)   (算子的输入是一个时间点, 输出是下一个时间点)
  skip(t) = Linear(x_t1)*exp(-lam*t_norm): 只作用于训练窗口内 (t1..t_k) 的"输出";
           外推区 (t_k+1..t_T) 不加 skip, 纯算子; skip 不参与状态轨迹.

三个场景 (训练只用 t1..t_k; 尾部 n_test 个点评分; 三个场景都输出全轨迹):
  rec_t1 : 起点 = 真 t1, 之后每步输入自己的预测        (纯 recursive)
  rec_tk : 同 rec_t1 滚到 t_k, 在 t_k 把状态换成真值, 再滚尾部 (作者 CV3 / forecasting)
  iter   : 每步输入真值, 算子固定                      (算子能力上界; 作者 tp1_iter 口径)

[FIX 2026-09-11]
  FIX-1 scaler 只用训练窗口(t1..t_k)的训练基因型拟合 (不用 t_k+1.. 的未来表型)
  FIX-2 skip 时间轴训练/评估统一 (都用全轴 t_norm)
  FIX-3 全轨迹 PCC/MSE 按 inner-5 集成
[FIX 2026-09-12] 口径对齐
  * 外推区不加 skip
  * rec_tk 改为 "滚到 t_k 后注入真 t_k 再滚尾部"; iter 改为全轨迹每步喂真值
协议: fold1-5 train / fold6 test; 内 5-fold; rank2 + skip + affine drift.
"""
import math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from pathlib import Path
import warnings, time, argparse
warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANK = 2; HID = 128
LR = 1e-3; WD = 1e-4
BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "outputs" / "reproduce" / "knode_extrap_maize_arab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def grm_dim_for(dataset):
    return 50 if dataset == "maize" else 30

# ---------------- 主方法模型 (rank2 + skip + affine) ----------------
class HyperSkipODEModel(nn.Module):
    def __init__(self, P, COND_IN, rank=2, hid=128, use_skip=True, drift="affine"):
        super().__init__()
        self.P = P; self.use_skip = use_skip; self.drift = drift
        self.hyper_enc = nn.Sequential(nn.Linear(COND_IN, hid), nn.Tanh())
        self.hyper_U = nn.Linear(hid, P * rank)
        self.hyper_V = nn.Linear(hid, rank * P)
        if use_skip:
            self.skip_proj = nn.Linear(P, P)
            self.log_lambda = nn.Parameter(torch.tensor(0.0))
        if drift == "affine":
            self.hyper_b = nn.Linear(hid, P)
        self._U_dyn = self._V_dyn = self._b_dyn = None

    def set_z(self, grm_pca, y1):
        cond = torch.cat([grm_pca, y1], dim=-1)
        B = grm_pca.shape[0]
        h = self.hyper_enc(cond)
        self._U_dyn = self.hyper_U(h).view(B, self.P, RANK)
        self._V_dyn = self.hyper_V(h).view(B, RANK, self.P)
        if self.drift == "affine":
            self._b_dyn = self.hyper_b(h).view(B, self.P)

    def forward(self, t, h):
        lin = torch.bmm(torch.bmm(h.unsqueeze(1), self._V_dyn.transpose(1, 2)),
                        self._U_dyn.transpose(1, 2)).squeeze(1)
        return lin + self._b_dyn if self.drift == "affine" else lin

    def compute_skip(self, h0, t_norm):
        proj = self.skip_proj(h0)
        decay = torch.exp(-torch.exp(self.log_lambda) * t_norm)
        return proj.unsqueeze(1) * decay.view(1, -1, 1)


def euler_integrate(model, h0, ts):
    B, D = h0.shape; T_len = len(ts)
    traj = torch.zeros(B, T_len, D, device=h0.device, dtype=h0.dtype); traj[:, 0] = h0
    h = h0
    for i in range(1, T_len):
        dt = max(ts[i].item() - ts[i - 1].item(), 1e-6)
        h = h + model(None, h) * dt; traj[:, i] = h
    return traj


def iter_forward(model, y, ts):
    B, T_len, P = y.shape
    out = torch.zeros_like(y); out[:, 0, :] = y[:, 0, :]
    for t in range(T_len - 1):
        dt = max(ts[t + 1].item() - ts[t].item(), 1e-6)
        out[:, t + 1, :] = y[:, t, :] + model(None, y[:, t, :]) * dt
    return out


def load_data(dataset, grm_dim):
    if dataset == "maize":
        pheno = pd.read_csv(BASE / "data" / "maize_magic" / "maize_trait_BLUP_50_NArmv_traits_afterClustering.csv")
        cv = pd.read_csv(BASE / "data" / "maize_magic" / "cross_validation_folds_w_validation_set_maize.csv", index_col=0)
        grm = pd.read_csv(BASE / "data" / "maize_magic" / "GRM_330x330_methodA_from552.csv", index_col=0)
    else:
        pheno = pd.read_csv(BASE / "data" / "arabidopsis" / "BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv")
        cv = pd.read_csv(BASE / "data" / "arabidopsis" / "cross_validation_folds_w_validation_set_arabidopsis.csv", index_col=0)
        grm = pd.read_csv(BASE / "data" / "arabidopsis" / "arabidopsis_GRM_382.csv", index_col=0)
    meta = ["bio_ID", "accession", "DAS"]
    traits = [c for c in pheno.columns if c not in meta and not c.startswith("Unnamed") and c != ""]
    traits = [t for t in traits if pheno[t].std() > 1e-12]
    P = len(traits)
    times = sorted(pheno["DAS"].unique())
    T = len(times)
    common = sorted(set(pheno["bio_ID"].astype(str)) & set(cv.index))
    grm.index = grm.index.astype(str); grm.columns = grm.columns.astype(str)
    grm = grm.loc[common, common].values.astype(np.float64)
    pca = PCA(n_components=grm_dim)
    grm_pca = pca.fit_transform(grm).astype(np.float32)
    Y = np.full((len(common), T, P), np.nan, dtype=np.float64)
    for pi, tr in enumerate(traits):
        sub = pheno[["bio_ID", "DAS", tr]].copy(); sub["bio_ID"] = sub["bio_ID"].astype(str)
        Y[:, :, pi] = sub.pivot(index="bio_ID", columns="DAS", values=tr).loc[common].values
    assert not np.any(np.isnan(Y))
    tn = (np.array(times, dtype=np.float64) - times[0]) / (times[-1] - times[0] + 1e-8)
    return dict(P=P, T=T, N=len(common), traits=traits, times=np.array(times), tn=tn,
                grm_pca=grm_pca, Y=Y, cv=cv)


def train_model(D, foldset, n_test, epochs, dataset, inner=5):
    grm_pca, Y, cv = D["grm_pca"], D["Y"], D["cv"]
    P, T, N = D["P"], D["T"], D["N"]
    col = cv.iloc[:, foldset-1].values
    tr = np.where((col >= 1) & (col <= 5))[0]; te = np.where(col == 6)[0]
    in_tr = tr[col[tr] != inner]; in_va = tr[col[tr] == inner]   # 折内: 4/5 训练 + 1/5 早停
    tr_len = T - n_test
    # [FIX-1] scaler 只用训练窗口(t1..t_k)的训练基因型拟合; 测试点 t_k+1.. 允许超出 [0,1] (不 clip)
    tr_min = np.nanmin(Y[tr, :tr_len], axis=(0, 1))
    tr_max = np.nanmax(Y[tr, :tr_len], axis=(0, 1))
    tr_rng = np.maximum(tr_max - tr_min, 1e-8)
    Ys_in = (Y[in_tr] - tr_min) / tr_rng
    Ys_va = (Y[in_va] - tr_min) / tr_rng
    Ys_te = (Y[te] - tr_min) / tr_rng   # 统一: 训练折 scaler
    g_in = torch.tensor(grm_pca[in_tr], device=DEVICE); g_va = torch.tensor(grm_pca[in_va], device=DEVICE)
    g_te = torch.tensor(grm_pca[te], device=DEVICE)
    ts_all = torch.tensor(D["tn"], dtype=torch.float32, device=DEVICE)
    ts_tr = ts_all[:tr_len]          # train window times
    ts_te = ts_all[tr_len - 1:]      # tail from t_k to end
    y_in = torch.tensor(Ys_in[:, :tr_len, :].astype(np.float32), device=DEVICE)
    y_va = torch.tensor(Ys_va[:, :tr_len, :].astype(np.float32), device=DEVICE)
    y_te = torch.tensor(Ys_te.astype(np.float32), device=DEVICE)
    # 归一化时间 (用于 skip decay)
    def tnorm(ts):
        return (ts - ts[0]) / (ts[-1] - ts[0] + 1e-8)
    # [FIX-2] skip 时间轴统一: 训练与评估都用全轴 t_norm
    tn_all = tnorm(ts_all)
    tn_tr = tn_all[:tr_len]
    tn_te = tn_all[tr_len - 1:]

    model = HyperSkipODEModel(P, grm_pca.shape[1] + P, RANK, HID, use_skip=True, drift="affine").to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best = float("inf"); best_state = None
    for ep in range(epochs):
        model.train(); model.set_z(g_in, y_in[:, 0, :])
        out = euler_integrate(model, y_in[:, 0, :], ts_tr) + model.compute_skip(y_in[:, 0, :], tn_tr)
        loss = F.mse_loss(out, y_in)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % 100 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                model.set_z(g_va, y_va[:, 0, :])
                ov = euler_integrate(model, y_va[:, 0, :], ts_tr) + model.compute_skip(y_va[:, 0, :], tn_tr)
                vl = F.mse_loss(ov[:, 1:, :], y_va[:, 1:, :]).item()
            if best_state is None or (not math.isnan(vl) and vl < best):
                best = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    return model, D, te, ts_all, ts_tr, ts_te, tn_tr, tn_te, tn_all, Ys_te, g_te, y_te, tr_len


def pcc_per_time(pred, true):
    res = []
    for t in range(pred.shape[1]):
        for pi in range(pred.shape[2]):
            r, _ = pearsonr(true[:, t, pi], pred[:, t, pi])
            res.append(r)
    return float(np.nanmean(res))


def _safe_pcc(x, y):
    """NaN 安全 PCC; 常数或样本不足返回 NaN."""
    x = np.asarray(x, dtype=np.float64).ravel(); y = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(pearsonr(x, y)[0])


def mean_snapshot_pcc(true, pred):
    """(N, K): 每列跨 N 的 PCC, 再跨 K 平均 -> 作者口径 snapshot PCC."""
    t = np.asarray(true, dtype=np.float64); p = np.asarray(pred, dtype=np.float64)
    if t.ndim == 1:
        t, p = t[:, None], p[:, None]
    return float(np.nanmean([_safe_pcc(t[:, k], p[:, k]) for k in range(t.shape[1])]))


def mean_longitudinal_pcc(true, pred):
    """(N, K): 每行跨 K 的 PCC, 再跨 N 平均 -> 论文口径 longitudinal PCC."""
    t = np.asarray(true, dtype=np.float64); p = np.asarray(pred, dtype=np.float64)
    if t.ndim == 1:
        t, p = t[None, :], p[None, :]
    return float(np.nanmean([_safe_pcc(t[i, :], p[i, :]) for i in range(t.shape[0])]))


def pooled_metrics(true, pred):
    """同一归一化空间 pooled: mse=mean((t-p)^2), nmse=mse/var(t), nrmse=sqrt(nmse), r2=1-nmse."""
    t = np.asarray(true, dtype=np.float64); p = np.asarray(pred, dtype=np.float64)
    mse = float(np.nanmean((t - p) ** 2))
    var = float(np.nanvar(t))
    nmse = float(mse / var) if var > 1e-12 else np.nan
    nrmse = float(np.sqrt(nmse)) if np.isfinite(nmse) else np.nan
    r2 = float(1.0 - nmse) if np.isfinite(nmse) else np.nan
    if t.ndim == 3:      # (N, T, P)
        snap = float(np.nanmean([[_safe_pcc(t[:, j, k], p[:, j, k])
                                  for j in range(t.shape[1])] for k in range(t.shape[2])]))
        long = float(np.nanmean([[_safe_pcc(t[i, :, k], p[i, :, k])
                                  for i in range(t.shape[0])] for k in range(t.shape[2])]))
    elif t.ndim == 2:    # (N, K)
        snap, long = mean_snapshot_pcc(t, p), mean_longitudinal_pcc(t, p)
    else:
        snap, long = _safe_pcc(t, p), np.nan
    return dict(pcc=_safe_pcc(t, p), pcc_snapshot=snap, pcc_longitudinal=long,
                mse=mse, nmse=nmse, nrmse=nrmse, r2=r2, n=int(t.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="maize", choices=["maize", "arab"])
    ap.add_argument("--n_test", type=int, default=5)
    ap.add_argument("--foldsets", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    foldsets = [int(x) for x in a.foldsets.split(",")]
    grm_dim = grm_dim_for(a.dataset)
    D = load_data(a.dataset, grm_dim)
    P, T = D["P"], D["T"]
    n_test = int(min(a.n_test, T - 2))
    score_idx = list(range(T - n_test, T))
    print(f"[DEVICE={DEVICE}] dataset={a.dataset} P={P} T={T} n_test={n_test} grm_dim={grm_dim} foldsets={foldsets} epochs={a.epochs}", flush=True)

    rows = []          # legacy: per (foldset, scenario, time) snapshot PCC
    full_rows = []
    full_long = []     # [FIX-3] 全轨迹逐点 (rec_t1 / t1_iter)
    full_trait = []    # 逐性状 x 逐时间点 (供箱图)
    overall_rows = []  # 三层之一: overall (pooled)
    time_rows = []     # 三层之二: by_time
    trait_rows = []    # 三层之三: by_trait
    for foldset in foldsets:
        acc = {"rec_t1": [], "rec_tk": [], "iter": []}
        acc_full = {"rec_t1": [], "rec_tk": [], "iter": []}   # 全轨迹按 inner 累加
        for inner in range(1, 6):
            model, DD, te, ts_all, ts_tr, ts_te, tn_tr, tn_te, tn_all, Ys_te, g_te, y_te, tr_len = \
                train_model(D, foldset, n_test, a.epochs, a.dataset, inner=inner)
            with torch.no_grad():
                # 算子锚点恒为 [grm_pca, x_t1]; skip 只作用于训练窗口内的输出
                y1 = y_te[:, 0, :]
                model.set_z(g_te, y1)
                skip_full = model.compute_skip(y1, tn_all)              # (B,T,P)
                def with_skip(ode_traj):
                    out = ode_traj.clone()
                    out[:, :tr_len, :] = out[:, :tr_len, :] + skip_full[:, :tr_len, :]
                    return out
                # --- rec_t1: 真 t1 起步, 每步输入自己的预测 ---
                p_t1 = with_skip(euler_integrate(model, y1, ts_all))
                # --- rec_tk: 同 rec_t1 滚到 t_k, 在 t_k 注入真值, 再滚尾部 (尾部纯算子) ---
                h_k = y_te[:, tr_len - 1, :]                            # 真 t_k
                ode_tk = euler_integrate(model, h_k, ts_all[tr_len - 1:])  # [0]=真 t_k, [1:]=尾部
                p_tk = euler_integrate(model, y1, ts_all).clone()
                p_tk[:, tr_len:, :] = ode_tk[:, 1:, :]
                p_tk = with_skip(p_tk)
                # --- iter: 每步输入真值 (算子固定) ---
                p_it = with_skip(iter_forward(model, y_te, ts_all))
                for _scen, _full in (("rec_t1", p_t1), ("rec_tk", p_tk), ("iter", p_it)):
                    acc[_scen].append(_full[:, score_idx, :])
                    acc_full[_scen].append(_full.detach().cpu().numpy())
        preds = {k: torch.stack(v, 0).mean(0) for k, v in acc.items()}          # inner-5 集成 (尾部)
        full_ens = {k: np.stack(v, 0).mean(0) for k, v in acc_full.items()}     # [FIX-3] 全轨迹集成
        # 全轨迹逐点 PCC/MSE: rec_t1 = 从真 t1 起纯递推, t1_iter = 每步喂真值
        try:
            for scen_f, arr in full_ens.items():
                for j in range(0, T):
                    pr = arr[:, j, :]; tr_ = Ys_te[:, j, :]
                    full_long.append(dict(dataset=a.dataset, foldset=foldset, scenario=scen_f,
                                          time=int(D["times"][j]),
                                          pcc=float(np.nanmean([_safe_pcc(tr_[:, k], pr[:, k])
                                                                for k in range(tr_.shape[1])])),
                                          mse=float(np.nanmean((tr_ - pr) ** 2))))
                for j in range(0, T):
                    for pidx, tname in enumerate(D["traits"]):
                        prp = arr[:, j, pidx]; trp = Ys_te[:, j, pidx]
                        full_trait.append(dict(dataset=a.dataset, foldset=foldset, scenario=scen_f,
                                               trait=tname, time=int(D["times"][j]),
                                               pcc=_safe_pcc(trp, prp),
                                               mse=float(np.nanmean((trp - prp) ** 2))))
            arr = full_ens["rec_t1"]   # 兼容旧宽表
            fp, fm = {}, {}
            for j in range(1, T):
                pr = arr[:, j, :]; tr_ = Ys_te[:, j, :]
                fp[int(D["times"][j])] = float(np.nanmean([_safe_pcc(tr_[:, k], pr[:, k]) for k in range(tr_.shape[1])]))
                fm[int(D["times"][j])] = float(np.nanmean((tr_ - pr) ** 2))
            full_rows.append(dict(dataset=a.dataset, foldset=foldset,
                                  **{f"PCC{k}": v for k, v in fp.items()},
                                  **{f"MSE{k}": v for k, v in fm.items()}))
        except Exception as _e:
            print("full-pcc skip:", _e)
        true_tail = np.asarray(Ys_te[:, score_idx, :], dtype=np.float64)
        for scen in ["rec_t1", "rec_tk", "iter"]:
            pr = preds[scen].cpu().numpy().astype(np.float64)
            tr_ = true_tail
            # overall: pooled over (genotypes x tail timepoints x traits)
            mo = pooled_metrics(tr_, pr)
            overall_rows.append(dict(dataset=a.dataset, foldset=foldset, scenario=scen, **mo))
            # by_time: 每个尾部时点 (snapshot = 跨基因型 PCC, 再跨性状平均)
            for j, tp in enumerate(D["times"][score_idx]):
                mt = pooled_metrics(tr_[:, j, :], pr[:, j, :])
                mt["pcc_longitudinal"] = np.nan
                time_rows.append(dict(dataset=a.dataset, foldset=foldset, scenario=scen,
                                      time=int(tp), **mt))
                rows.append(dict(dataset=a.dataset, foldset=foldset, scenario=scen,
                                 time=int(tp), pcc=float(mt["pcc_snapshot"])))
            # by_trait
            for pidx, tname in enumerate(D["traits"]):
                trp, prp = tr_[:, :, pidx], pr[:, :, pidx]
                mt = pooled_metrics(trp, prp)
                trait_rows.append(dict(dataset=a.dataset, foldset=foldset, scenario=scen,
                                       trait=tname, **mt))
            print("  foldset %d %s: PCC=%.4f snapPCC=%.4f longPCC=%.4f MSE=%.6f NMSE=%.4f NRMSE=%.4f R2=%.4f" % (
                foldset, scen, mo["pcc"], mo["pcc_snapshot"], mo["pcc_longitudinal"],
                mo["mse"], mo["nmse"], mo["nrmse"], mo["r2"]), flush=True)

    tag = a.tag or f"v2_f{len(foldsets)}_nt{n_test}"
    pd.DataFrame(full_rows).to_csv(
        OUT_DIR / f"knode_extrap_{a.dataset}_fulltraj_pcc{('_' + a.tag) if a.tag else ''}.csv", index=False)
    df = pd.DataFrame(rows)
    out = OUT_DIR / f"knode_extrap_{a.dataset}_{tag}.csv"
    df.to_csv(out, index=False)
    fl = pd.DataFrame(full_long)
    fl.to_csv(OUT_DIR / f"knode_extrap_{a.dataset}_fulltraj_{tag}.csv", index=False)
    pd.DataFrame(full_trait).to_csv(
        OUT_DIR / f"knode_extrap_{a.dataset}_fulltraj_bytrait_{tag}.csv", index=False)
    if not fl.empty:
        fl.groupby(["dataset", "scenario", "time"])[["pcc", "mse"]].agg(["mean", "std"]).round(6).to_csv(
            OUT_DIR / f"knode_extrap_{a.dataset}_fulltraj_summary_{tag}.csv")

    d_overall, d_time, d_trait = pd.DataFrame(overall_rows), pd.DataFrame(time_rows), pd.DataFrame(trait_rows)
    d_overall.to_csv(OUT_DIR / f"knode_extrap_{a.dataset}_{tag}_overall.csv", index=False)
    d_time.to_csv(OUT_DIR / f"knode_extrap_{a.dataset}_{tag}_by_time.csv", index=False)
    d_trait.to_csv(OUT_DIR / f"knode_extrap_{a.dataset}_{tag}_by_trait.csv", index=False)

    METRICS = ["pcc", "pcc_snapshot", "pcc_longitudinal", "mse", "nmse", "nrmse", "r2"]

    def _summ(d, keys):
        g = d.groupby(keys, dropna=False)[METRICS].agg(["mean", "std"])
        g.columns = [f"{m}_{s}" for m, s in g.columns]
        g["n_foldsets"] = d.groupby(keys, dropna=False)["foldset"].nunique()
        return g.reset_index()

    s_overall = _summ(d_overall, ["dataset", "scenario"]); s_overall["level"] = "overall"
    s_time = _summ(d_time, ["dataset", "scenario", "time"]); s_time["level"] = "by_time"
    s_trait = _summ(d_trait, ["dataset", "scenario", "trait"]); s_trait["level"] = "by_trait"
    summ = pd.concat([s_overall, s_time, s_trait], ignore_index=True)
    summ.to_csv(OUT_DIR / f"knode_extrap_{a.dataset}_{tag}_summary.csv", index=False)

    import hashlib, datetime
    meta = dict(dataset=a.dataset, P=P, T=T, n_test=n_test, grm_dim=grm_dim,
                foldsets=",".join(map(str, foldsets)), n_foldsets=len(foldsets), epochs=a.epochs,
                rank=RANK, hid=HID, lr=LR, wd=WD, inner_ensemble=5, device=str(DEVICE),
                scaler="train-fold per-trait min-max (val/test 不参与统计量)",
                metrics="pcc,mse,nmse,nrmse,r2 同归一化空间; snapshot=跨基因型, longitudinal=跨时点",
                timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                script=__file__, script_md5=hashlib.md5(open(__file__, "rb").read()).hexdigest())
    pd.DataFrame([{"key": k, "value": v} for k, v in meta.items()]).to_csv(
        OUT_DIR / f"knode_extrap_{a.dataset}_{tag}_meta.csv", index=False)

    print("=== overall (mean over foldsets) ===", flush=True)
    cols = ["scenario", "pcc_mean", "pcc_snapshot_mean", "pcc_longitudinal_mean",
            "mse_mean", "nmse_mean", "nrmse_mean", "r2_mean", "n_foldsets"]
    print(s_overall[cols].round(4).to_string(index=False), flush=True)
    print("saved ->", out, flush=True)
    print("saved ->", OUT_DIR / f"knode_extrap_{a.dataset}_{tag}_summary.csv", flush=True)


if __name__ == "__main__":
    main()
