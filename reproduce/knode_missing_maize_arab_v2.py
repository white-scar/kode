# -*- coding: utf-8 -*-
"""knode_missing_maize_arab_v2.py - KNode(主方法 no-skip + affine drift, rank=2) 缺失值实验, 十折, 拟南芥 dim30.

场景: 玉米 drop1/drop2/drop3/delblock; 拟南芥 drop_last1/2/3, gap_mid.
训练: 从 t1 整轨迹 rollout, loss 只算 kept 时间点(masked loss).
预测: rec(从 t1 全滚) / iter(kept 用真值, extra 自回归穿过), 评分 extra 时间点.
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
OUT = BASE / "outputs" / "reproduce" / "knode_missing_maize_arab"
OUT.mkdir(parents=True, exist_ok=True)

def grm_dim_for(dataset):
    return 50 if dataset == "maize" else 30

class HyperODEModel(nn.Module):
    def __init__(self, P, COND_IN, rank=2, hid=128, drift="affine"):
        super().__init__()
        self.P = P; self.drift = drift
        self.hyper_enc = nn.Sequential(nn.Linear(COND_IN, hid), nn.Tanh())
        self.hyper_U = nn.Linear(hid, P * rank)
        self.hyper_V = nn.Linear(hid, rank * P)
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


def integrate(model, h0, ts):
    B, D = h0.shape; T = len(ts)
    traj = torch.zeros(B, T, D, device=h0.device, dtype=h0.dtype); traj[:, 0] = h0
    h = h0
    for i in range(1, T):
        dt = max(ts[i].item() - ts[i - 1].item(), 1e-6)
        h = h + model(None, h) * dt; traj[:, i] = h
    return traj


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


def scenario_masks(dataset):
    if dataset == "maize":
        blocks = [[15,16,17,18,19],[22,23,24,25,26],[29,30,31,32,33],[36,37,38,39,40],[43,44,45,46,47]]
        cfg = {"drop1": dict(keep_per=4, extra=[47]), "drop2": dict(keep_per=3, extra=[46,47]),
               "drop3": dict(keep_per=2, extra=[45,46,47]), "delblock": dict(keep_blocks=4, extra=[43,44,45,46,47]),
               # 挖 t6-10 + t16-20 (中间两段), 外推 t21-25
               "hole": dict(kept=[15,16,17,18,19, 29,30,31,32,33],
                            extra=[22,23,24,25,26, 36,37,38,39,40, 43,44,45,46,47])}
        scen = {}
        for name, c in cfg.items():
            if "kept" in c:
                scen[name] = (c["kept"], c["extra"]); continue
            kept = [x for b in blocks[:c["keep_blocks"]] for x in b] if "keep_blocks" in c else \
                   [x for b in blocks for x in b[:c["keep_per"]]]
            scen[name] = (kept, c["extra"])
        return scen
    else:
        return {"drop_last1": ([t for t in range(8,21)], [21]),
                "drop_last2": ([t for t in range(8,20)], [20,21]),
                "drop_last3": ([t for t in range(8,19)], [19,20,21]),
                "gap_mid": ([8,9,10,11,17,18,19,20,21], [12,13,14,15,16])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="maize", choices=["maize", "arab"])
    ap.add_argument("--foldsets", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--tag", default="")
    ap.add_argument("--scen", default="", help="只跑指定场景, 逗号分隔; 空=全部")
    a = ap.parse_args()
    foldsets = [int(x) for x in a.foldsets.split(",") if x.strip()]
    grm_dim = grm_dim_for(a.dataset)
    D = load_data(a.dataset, grm_dim)
    P, T, N = D["P"], D["T"], D["N"]
    times = D["times"]; tn = D["tn"]; grm_pca = D["grm_pca"]; Y = D["Y"]; cv = D["cv"]
    scen = scenario_masks(a.dataset)
    if a.scen.strip():
        keep = [x.strip() for x in a.scen.split(',') if x.strip()]
        scen = {k: v for k, v in scen.items() if k in keep}
    ts = torch.tensor(tn, dtype=torch.float32, device=DEVICE)
    print(f"[DEVICE={DEVICE}] dataset={a.dataset} P={P} T={T} grm_dim={grm_dim} foldsets={foldsets} epochs={a.epochs}", flush=True)

    rows = []
    for foldset in foldsets:
        col = cv.iloc[:, foldset - 1].values
        tr_pos = np.where((col >= 1) & (col <= 5))[0]; te_pos = np.where(col == 6)[0]
        in_tr = tr_pos[(col[tr_pos] >= 1) & (col[tr_pos] <= 4)]; in_va = tr_pos[col[tr_pos] == 5]
        tr_min = np.nanmin(Y[tr_pos], axis=(0, 1)); tr_rng = np.maximum(np.nanmax(Y[tr_pos], axis=(0, 1)) - tr_min, 1e-8)
        Ys_in = (Y[in_tr] - tr_min) / tr_rng; Ys_va = (Y[in_va] - tr_min) / tr_rng; Ys_te = (Y[te_pos] - tr_min) / tr_rng   # 统一: 训练折 scaler
        g_in = torch.tensor(grm_pca[in_tr], device=DEVICE); g_va = torch.tensor(grm_pca[in_va], device=DEVICE)
        g_te = torch.tensor(grm_pca[te_pos], device=DEVICE)

        for sname, (kept, extra) in scen.items():
            kept_idx = [i for i, t in enumerate(times) if t in kept]
            extra_idx = [i for i, t in enumerate(times) if t in extra]
            pr_list = []; pi_list = []
            for inner in range(1, 6):          # 折内 5 折: 4/5 训练 + 1/5 早停 -> 集成
                i_in = tr_pos[col[tr_pos] != inner]; i_va = tr_pos[col[tr_pos] == inner]
                Ys_i = (Y[i_in] - tr_min) / tr_rng; Ys_v = (Y[i_va] - tr_min) / tr_rng
                g_i = torch.tensor(grm_pca[i_in], device=DEVICE); g_v = torch.tensor(grm_pca[i_va], device=DEVICE)
                model = HyperODEModel(P, grm_dim + P, RANK, HID, drift="affine").to(DEVICE)
                opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
                y_in = torch.tensor(Ys_i.astype(np.float32), device=DEVICE)
                y_va = torch.tensor(Ys_v.astype(np.float32), device=DEVICE)
                best = float("inf"); best_state = None
                for ep in range(a.epochs):
                    model.train(); model.set_z(g_i, y_in[:, 0, :])
                    out = integrate(model, y_in[:, 0, :], ts)
                    loss = F.mse_loss(out[:, kept_idx, :], y_in[:, kept_idx, :])
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    if ep % 100 == 0 or ep == a.epochs - 1:
                        model.eval()
                        with torch.no_grad():
                            model.set_z(g_v, y_va[:, 0, :])
                            ov = integrate(model, y_va[:, 0, :], ts)
                            vl = F.mse_loss(ov[:, kept_idx, :], y_va[:, kept_idx, :]).item()
                        if best_state is None or (not math.isnan(vl) and vl < best):
                            best = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(best_state); model.eval()
                with torch.no_grad():
                    y_te = torch.tensor(Ys_te.astype(np.float32), device=DEVICE)
                    model.set_z(g_te, y_te[:, 0, :])
                    pred_rec = integrate(model, y_te[:, 0, :], ts)
                    pred_iter = torch.zeros_like(y_te); pred_iter[:, 0, :] = y_te[:, 0, :]
                    for i in range(1, T):
                        dt = max(ts[i].item() - ts[i - 1].item(), 1e-6)
                        src = y_te[:, i - 1, :] if (i - 1) in kept_idx else pred_iter[:, i - 1, :]
                        pred_iter[:, i, :] = src + model(None, src) * dt
                pr_list.append(pred_rec); pi_list.append(pred_iter)
            pred_rec = torch.stack(pr_list, 0).mean(0)     # inner-5 集成
            pred_iter = torch.stack(pi_list, 0).mean(0)
            true_te = Y[te_pos]
            for mode, pred in [("rec", pred_rec.cpu().numpy()), ("iter", pred_iter.cpu().numpy())]:
                pred_raw = pred * tr_rng + tr_min   # 训练折 scaler 还原
                for j in extra_idx:
                    vals = [pearsonr(true_te[:, j, p], pred_raw[:, j, p])[0] for p in range(P)]
                    rows.append(dict(dataset=a.dataset, foldset=foldset, scenario=sname, mode=mode,
                                     time=int(times[j]), pcc=float(np.nanmean(vals))))
            print(f"  fold{foldset} {sname} done", flush=True)

    df = pd.DataFrame(rows)
    tag = a.tag or f"{a.dataset}_v2_f{len(foldsets)}"
    df.to_csv(OUT / f"knode_missing_{tag}.csv", index=False)
    print(df.groupby(["scenario", "mode"])["pcc"].mean().round(4).to_string(), flush=True)
    print("saved ->", OUT / f"knode_missing_{tag}.csv", flush=True)


if __name__ == "__main__":
    main()
