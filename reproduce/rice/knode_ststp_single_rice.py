# -*- coding: utf-8 -*-
"""knode_ststp_single_rice.py - ST-STP 单性状全覆盖 (水稻 20 个性状各训 P=1) (不使用 GRM) 动态多步预测实验 E1/E2/E3

依据《基序未来：TP1-only与单/少性状动态预测实验实施计划》:
  E1 (--mode all)    : 只用 TP1 的全部 P 个性状 (不输入 GRM) -> 预测同 P 性状未来 T-1 个时点
  E2 (--mode single) : 只用 1 个目标性状的 TP1               -> 预测该性状未来轨迹 (新建 P=1 模型)
  E3 (--mode few)    : 只用 3-5 个性状的 TP1                 -> 预测这些性状未来轨迹 (新建 P=3-5 模型)

与 knode_dynamic_baselines_full.py 的关系: **完全独立的新脚本**, 旧脚本/旧输出一律不改。
本脚本 **不读取任何 GRM/SNP 文件** (GRM 彻底解耦)。

协议 (与旧脚本一致):
  * 每个 fold-set k: folds1-5 训练, fold6 测试
  * 折内 5 折 (inner): 4/5 训练 + 1/5 早停; 测试预测 = 5 个 inner 模型平均
  * 归一化: 只用训练折的 per-trait min/max (仅所选性状), 套用到训练/验证/测试
  * 固定 seed = 42 (不做多随机种子实验; 旧脚本里的 *_seed 实为 inner-fold 预测)
  * 指标在归一化空间计算; 评价排除 t0

输出 (--outdir, 默认 <root>/outputs/reproduce/knode_tp1only_maize/<mode>[_tag]):
  metrics_by_trait_time_per_iter.csv  逐 iteration 主表 (含 fam/time/trait + 9 指标)
  metrics_by_trait_time.csv           iteration 平均主表 (11 列 + nan_reason)
  metrics_by_trait_time_std.csv       iteration 间各指标标准差 (ddof=0, 已标注)
  metrics_overall.csv                 每 fam 总体指标 (全部共同性状 x t0 后全部时点)
  predictions_long.csv.gz             iteration,fam,bio_id,time,trait,y_true,y_pred
  run_config.json / train.log

用法: cd <root>/reproduce && python knode_tp1only_dynamic.py --mode all --iters 0,1,2 --epochs 1000
冒烟: python knode_tp1only_dynamic.py --mode all --smoke
"""
import argparse, json, os, sys, time, gzip
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr

# ---------------- 路径 / 常量 ----------------
ROOT = Path(os.environ.get("KNODE_ROOT", "..")).resolve()
DATA_DIR = ROOT / "data" / "rice_salinity"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3; WD = 1e-4
SEED = 42                     # 固定种子 (计划书: 不做多随机种子)
META = ["bio_ID", "accession", "DAS"]

# ---------------- 数据 (完全不含 GRM) ----------------
def load_data():
    pheno = pd.read_csv(DATA_DIR / "rice_indica_control_traits.csv", index_col=0)
    pheno = pheno[pheno["DAS"] >= 30].copy()   # 与项目其余水稻结果同口径 (DAS 30-42, T=13)
    cv_raw = pd.read_csv(DATA_DIR / "rice_cv_10fold.csv", index_col=0)
    cv_raw.index = cv_raw.index.astype(str)   # rice 的索引会被读成整数, 统一转字符串以便 loc[ids]
    trait_cols = [c for c in pheno.columns if c not in META
                  and not c.startswith("Unnamed") and c != ""]
    times = np.array(sorted(pheno["DAS"].unique()), dtype=np.float64)
    common_ids = sorted(set(pheno["bio_ID"].astype(str)) & set(cv_raw.index.astype(str)))
    Y = np.full((len(common_ids), len(times), len(trait_cols)), np.nan)
    for pi, tr in enumerate(trait_cols):
        sub = pheno[["bio_ID", "DAS", tr]].copy(); sub["bio_ID"] = sub["bio_ID"].astype(str)
        Y[:, :, pi] = sub.pivot(index="bio_ID", columns="DAS", values=tr).loc[common_ids].values
    assert not np.any(np.isnan(Y)), "phenotype 有缺失"
    tn = (times - times.min()) / (times.max() - times.min() + 1e-8)
    return dict(pheno=pheno, cv=cv_raw, trait_cols=trait_cols, times=times,
                ids=common_ids, Y=Y, T=len(times), P=len(trait_cols),
                tn=tn, times_t=torch.tensor(tn, dtype=torch.float32, device=DEVICE))

# ---------------- 模型族 (与旧脚本逐行一致) ----------------
def rk4_step(f, z, t0, t1):
    h = max(float(t1 - t0), 1e-6)
    k1 = f(z, t0); k2 = f(z + 0.5 * h * k1, t0 + 0.5 * h)
    k3 = f(z + 0.5 * h * k2, t0 + 0.5 * h); k4 = f(z + h * k3, t1)
    return z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

class GRUODEBayes(nn.Module):
    def __init__(self, P=50, COND_IN=100, D=96):
        super().__init__()
        self.enc = nn.Linear(COND_IN, D)
        self.W_z = nn.Linear(D, D); self.W_r = nn.Linear(D, D); self.W_h = nn.Linear(D, D)
        self.cell = nn.GRUCell(P, D); self.head = nn.Linear(D, P)
    def f_ode(self, h):
        z = torch.sigmoid(self.W_z(h)); r = torch.sigmoid(self.W_r(h))
        h_new = torch.tanh(self.W_h(r * h))
        return (1 - z) * (h_new - h)
    def forward(self, cond, ts):
        B = cond.shape[0]; h = torch.tanh(self.enc(cond))
        prev = torch.zeros(B, self.head.out_features, device=cond.device); preds = []
        for i in range(len(ts)):
            if i > 0:
                fz = lambda hh, t: self.f_ode(hh)
                h = rk4_step(fz, h, float(ts[i - 1]), float(ts[i]))
                h = self.cell(prev, h)
            yt = self.head(h); preds.append(yt); prev = yt.detach()
        return torch.stack(preds, dim=1)

class ARTransformer(nn.Module):
    def __init__(self, P=50, COND_IN=100, D=96, NH=2, FF=256, NL=2):
        super().__init__()
        self.time_emb = nn.Linear(1, D); self.pred_proj = nn.Linear(P, D)
        self.cond_proj = nn.Linear(COND_IN, D)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(D, NH, FF, batch_first=True), NL)
        self.head = nn.Linear(D, P)
    def forward(self, cond, ts, y_true=None, tf_prob=0.7):
        B = cond.shape[0]; c = self.cond_proj(cond).unsqueeze(1)
        prev = torch.zeros(B, self.head.out_features, device=cond.device)
        seq_tokens = torch.zeros(B, 0, self.time_emb.out_features, device=cond.device); preds = []
        for i in range(len(ts)):
            te = self.time_emb(ts[i].view(1, 1)).expand(B, -1).unsqueeze(1)
            step = te + self.pred_proj(prev).unsqueeze(1)
            seq_tokens = torch.cat([seq_tokens, step], dim=1); L = i + 1
            tgt_mask = torch.triu(torch.full((L, L), float("-inf"), device=cond.device), diagonal=1)
            out = self.dec(seq_tokens, c, tgt_mask=tgt_mask)[:, -1]
            yt = self.head(out); preds.append(yt)
            if self.training and y_true is not None:
                prev = y_true[:, i] if torch.rand(1).item() < tf_prob else yt.detach()
            else:
                prev = yt.detach()
        return torch.stack(preds, dim=1)

class ODERNNF(nn.Module):
    def __init__(self, P=50, COND_IN=100, D=96):
        super().__init__()
        self.enc = nn.Linear(COND_IN, D)
        self.f = nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Linear(D, D))
        self.cell = nn.LSTMCell(P, D); self.head = nn.Linear(D, P)
    def forward(self, cond, ts):
        B = cond.shape[0]; z = torch.tanh(self.enc(cond)); h = z; c = torch.zeros_like(z)
        prev = torch.zeros(B, self.head.out_features, device=cond.device); preds = []
        for i in range(len(ts)):
            if i > 0:
                fz = lambda zz, t: self.f(zz)
                z = rk4_step(fz, z, float(ts[i - 1]), float(ts[i]))
                h, c = self.cell(prev, (z, c)); z = h
            yt = self.head(z); preds.append(yt); prev = yt.detach()
        return torch.stack(preds, dim=1)

class STSTP(nn.Module):
    def __init__(self, P=50, COND_IN=100, T=25, D=128):
        super().__init__()
        self.P = P
        self.net = nn.Sequential(nn.Linear(COND_IN, D), nn.Tanh(), nn.Linear(D, T * P))
    def forward(self, cond, ts):
        B = cond.shape[0]
        return self.net(cond).view(B, len(ts), self.P)

class LSTMModel(nn.Module):
    def __init__(self, P=50, COND_IN=100, HID=128):
        super().__init__()
        self.fc_h = nn.Linear(COND_IN, HID); self.fc_c = nn.Linear(COND_IN, HID)
        self.cell = nn.LSTMCell(1, HID); self.head = nn.Linear(HID, P)
    def forward(self, cond, ts):
        B = cond.shape[0]
        h = torch.tanh(self.fc_h(cond)); c = torch.tanh(self.fc_c(cond)); outs = []
        for t in ts:
            h, c = self.cell(t.view(1).expand(B, 1), (h, c))
            outs.append(self.head(h))
        return torch.stack(outs, dim=1)

FAMS = ["ststp"]   # 本轮只跑 ST-STP

def build_model(fam, Pn, cond_in, Tn):
    if fam == "ststp":          return STSTP(Pn, cond_in, Tn).to(DEVICE)
    if fam == "lstm":           return LSTMModel(Pn, cond_in).to(DEVICE)
    if fam == "ar_transformer": return ARTransformer(Pn, cond_in).to(DEVICE)
    if fam == "ode_rnn":        return ODERNNF(Pn, cond_in).to(DEVICE)
    if fam == "gru_ode_bayes":  return GRUODEBayes(Pn, cond_in).to(DEVICE)
    raise ValueError(fam)

# ---------------- 训练 (TP1-only: cond = y1 本身, 不含 GRM) ----------------
def train_family(fam, Ys_tr, y1_tr, n_epochs, times_t, Ys_va=None, y1_va=None, log=print):
    torch.manual_seed(SEED); np.random.seed(SEED)
    B, Tn, Pn = Ys_tr.shape
    m = build_model(fam, Pn, Pn, Tn)          # COND_IN = Pn (只有 TP1)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    y_tr = torch.tensor(Ys_tr.astype(np.float32), device=DEVICE)
    y1_t = torch.tensor(y1_tr.astype(np.float32), device=DEVICE)
    if Ys_va is not None:
        y_va = torch.tensor(Ys_va.astype(np.float32), device=DEVICE)
        y1_va_t = torch.tensor(y1_va.astype(np.float32), device=DEVICE)
    else:
        y_va, y1_va_t = y_tr, y1_t
    best = None; best_loss = float("inf")
    for ep in range(n_epochs):
        m.train()
        pred = m(y1_t, times_t, y_true=y_tr, tf_prob=0.7) if fam == "ar_transformer" else m(y1_t, times_t)
        loss = F.mse_loss(pred, y_tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 50 == 0 or ep == n_epochs - 1:
            with torch.no_grad():
                m.eval(); pv = m(y1_va_t, times_t)
                vl = F.mse_loss(pv[:, 1:], y_va[:, 1:]).item()
            if not np.isnan(vl) and vl < best_loss:
                best_loss = vl; best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    if best is None:
        best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best); m.eval()
    return m

def predict_norm(m, fam, y1n, times_t):
    with torch.no_grad():
        y = torch.tensor(np.asarray(y1n, dtype=np.float32), device=DEVICE)
        return m(y, times_t).cpu().numpy()

# ---------------- 指标 (计划书 5.1 定义) ----------------
def cell_metrics(pred, true):
    """pred/true: (n_test,) 同一 trait/time."""
    out = dict(pcc=np.nan, mse=np.nan, RMSE=np.nan, R2=np.nan, NMSE=np.nan,
               NRMSE=np.nan, y_true=float(np.mean(true)), y_pred=float(np.mean(pred)), SD=np.nan,
               nan_reason="")
    if true.size < 3:
        out["nan_reason"] = "n_test<3"; return out
    sd_t = float(np.std(true, ddof=0))
    out["SD"] = sd_t                                        # 真实值标准差 (ddof=0)
    out["mse"] = float(np.mean((pred - true) ** 2))
    out["RMSE"] = float(np.sqrt(out["mse"]))
    sse = float(np.sum((pred - true) ** 2)); sst = float(np.sum((true - true.mean()) ** 2))
    if sst > 1e-12:
        out["R2"] = 1.0 - sse / sst
        out["NMSE"] = out["mse"] / (sst / true.size)         # = 1 - R2 (有效时)
    else:
        out["nan_reason"] = "SST=0"
    mt = float(np.mean(true))
    if abs(mt) > 1e-8:
        out["NRMSE"] = out["RMSE"] / abs(mt)
    else:
        out["nan_reason"] = (out["nan_reason"] + ";mean~0").strip(";")
    if sd_t > 1e-12 and float(np.std(pred, ddof=0)) > 1e-12:
        out["pcc"] = float(pearsonr(true, pred)[0])
    else:
        out["nan_reason"] = (out["nan_reason"] + ";zero_var").strip(";")
    return out

# ---------------- 一个 fold-set ----------------
def run_foldset(D, mode, sel_idx, fams, k, epochs, log):
    cv = D["cv"]; Y = D["Y"]; ids = D["ids"]; times_t = D["times_t"]
    sel = np.asarray(sel_idx); Pn = len(sel); times = D["times"]; T = D["T"]
    fc = cv.loc[ids].iloc[:, k].values.astype(int)
    tr = np.where((fc >= 1) & (fc <= 5))[0]; te = np.where(fc == 6)[0]
    Ysel = Y[:, :, sel]                                     # 只保留所选性状
    mn = np.nanmin(Ysel[tr], axis=(0, 1)); mx = np.nanmax(Ysel[tr], axis=(0, 1))
    rg = np.maximum(mx - mn, 1e-8)
    Ys_tr_all = ((Ysel[tr] - mn) / rg).astype(np.float32)
    y1_all = ((Ysel[:, 0, :] - mn) / rg).astype(np.float32)
    Y_te_n = ((Ysel[te] - mn) / rg).astype(np.float32)
    inner = fc[tr]
    te_ids = [ids[i] for i in te]
    t0 = time.time()
    log(f"[iter {k}] mode={mode} P={Pn} train={len(tr)} test={len(te)} | inner-5 ensemble | epochs={epochs}")
    rows, pred_rows = [], []
    for fam in fams:
        p_inner = []
        for j in range(1, 6):
            itr = inner != j; iva = inner == j
            m = train_family(fam, Ys_tr_all[itr], y1_all[tr][itr], epochs, times_t,
                             Ys_va=Ys_tr_all[iva], y1_va=y1_all[tr][iva], log=log)
            p_inner.append(predict_norm(m, fam, y1_all[te], times_t))
            if DEVICE.type == "cuda": torch.cuda.empty_cache()
        p_te = np.mean(p_inner, axis=0)                      # (n_te, T, P)
        log(f"  [iter {k}] {fam:15s} done  ({time.time()-t0:.0f}s)")
        for t in range(1, T):                                # 排除 t0
            for pi in range(Pn):
                mt = cell_metrics(p_te[:, t, pi], Y_te_n[:, t, pi])
                rows.append(dict(iteration=k, fam=fam, time=int(times[t]),
                                 trait=D["trait_cols"][sel[pi]], **mt))
                for ii, bid in enumerate(te_ids):
                    pred_rows.append((k, fam, bid, int(times[t]), D["trait_cols"][sel[pi]],
                                      float(Y_te_n[ii, t, pi]), float(p_te[ii, t, pi])))
    return rows, pred_rows, te_ids

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="TP1-only 动态预测实验 (E1/E2/E3)")
    ap.add_argument("--mode", required=True, choices=["all", "single", "few"])
    ap.add_argument("--traits", default="", help="single/few 模式的性状名, 逗号分隔")
    ap.add_argument("--fams", default=",".join(FAMS))
    ap.add_argument("--iters", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--smoke", action="store_true", help="冒烟: 强制 iters=0 且 epochs<=20")
    a = ap.parse_args()                                      # 严格 parse (未知参数直接报错)

    fams = [x.strip() for x in a.fams.split(",") if x.strip()]
    for f in fams:
        if f not in FAMS: ap.error(f"未知 fam: {f}")
    iters = [0] if a.smoke else [int(x) for x in a.iters.split(",") if x.strip()]
    epochs = min(a.epochs, 20) if a.smoke else a.epochs
    tag = a.tag or ("_smoke" if a.smoke else "")
    outdir = Path(a.outdir) if a.outdir else (ROOT / "outputs" / "reproduce" /
                                             "knode_ststp_single_rice" / f"{a.mode}{tag}")
    outdir.mkdir(parents=True, exist_ok=True)
    logp = outdir / "train.log"; logp.write_text("", encoding="utf-8")
    def log(msg):
        print(msg, flush=True)
        with open(logp, "a", encoding="utf-8") as fh: fh.write(msg + "\n")

    t_start = time.strftime("%Y-%m-%d %H:%M:%S")
    log(f"device={DEVICE} mode={a.mode} fams={fams} iters={iters} epochs={epochs} seed={SEED}")
    D = load_data()
    log(f"data: N={len(D['ids'])} T={D['T']} P_all={D['P']} times={D['times'].tolist()}")
    tc = D["trait_cols"]
    if a.mode == "all":
        sel_names = tc
    else:
        want = [x.strip() for x in a.traits.split(",") if x.strip()]
        if not want: ap.error("single/few 模式必须给 --traits")
        if a.mode == "single" and len(want) != 1: ap.error("single 模式只能给 1 个性状")
        if a.mode == "few" and not (3 <= len(want) <= 5): ap.error("few 模式需 3-5 个性状")
        miss = [w for w in want if w not in tc]
        if miss: ap.error(f"性状不在数据集中: {miss}")
        sel_names = want
    sel_idx = [tc.index(t) for t in sel_names]
    log(f"mode={a.mode}  P={len(sel_idx)}  性状={sel_names}")

    all_rows, all_preds, te_by_iter = [], [], {}
    for k in iters:
        rows, preds, te_ids = run_foldset(D, a.mode, sel_idx, fams, k, epochs, log)
        all_rows += rows; all_preds += preds; te_by_iter[k] = te_ids

    per_iter = pd.DataFrame(all_rows)
    per_iter.to_csv(outdir / "metrics_by_trait_time_per_iter.csv", index=False)
    keys = ["fam", "time", "trait"]; metrics = ["pcc", "mse", "RMSE", "R2", "NMSE", "NRMSE",
                                                "y_true", "y_pred", "SD"]
    main_tab = per_iter.groupby(keys, as_index=False)[metrics].mean()
    reasons = per_iter.groupby(keys)["nan_reason"].apply(
        lambda s: ";".join(sorted({x for x in s if isinstance(x, str) and x}))).reset_index()
    main_tab = main_tab.merge(reasons, on=keys, how="left")
    main_tab.to_csv(outdir / "metrics_by_trait_time.csv", index=False)
    (per_iter.groupby(keys, as_index=False)[metrics].std(ddof=0)
             .rename(columns={m: m + "_std" for m in metrics})
             .to_csv(outdir / "metrics_by_trait_time_std.csv", index=False))
    ov = per_iter.groupby("fam", as_index=False).agg(
        pcc=("pcc", "mean"), mse=("mse", "mean"), RMSE=("RMSE", "mean"),
        R2=("R2", "mean"), NMSE=("NMSE", "mean"), NRMSE=("NRMSE", "mean"),
        n_cells=("pcc", "size"), n_iter=("iteration", "nunique"))
    ov.to_csv(outdir / "metrics_overall.csv", index=False)
    with gzip.open(outdir / "predictions_long.csv.gz", "wt", newline="") as fh:
        fh.write("iteration,fam,bio_id,time,trait,y_true,y_pred\n")
        for r in all_preds:
            fh.write("%d,%s,%s,%d,%s,%.10g,%.10g\n" % r)
    (outdir / "run_config.json").write_text(json.dumps(dict(
        script="knode_ststp_single_rice.py", mode=a.mode, traits=sel_names, P=len(sel_idx),
        fams=fams, iters=iters, epochs=epochs, seed=SEED, cond_mode="tp1(no GRM)",
        dataset="rice_indica_control", data_file="rice_indica_control_traits.csv (DAS>=30)",
        cv_file="rice_cv_10fold.csv",
        normalization="train-fold per-trait min/max (selected traits only)",
        metrics_def=dict(pcc="Pearson r over test materials (per trait/time)",
                         mse="mean squared error", RMSE="sqrt(mse)",
                         R2="1 - SSE/SST (SST centered on test mean)",
                         NMSE="mse / Var(y_true), Var=SST/N  (= 1-R2)",
                         NRMSE="RMSE / |mean(y_true)|", y_true="mean of true values",
                         y_pred="mean of predicted values", SD="std of true values (ddof=0)"),
        eval_times="t0 excluded", t_start=t_start,
        t_end=time.strftime("%Y-%m-%d %H:%M:%S"),
        test_ids_by_iter=te_by_iter), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n=== overall ===")
    log(ov.to_string(index=False))
    log(f"\noutputs -> {outdir}")

if __name__ == "__main__":
    main()
