"""
knode_staygreen_selection.py — 持续优质(持绿型)植株的识别与模型对比 (用户 2026-08-17 设计)

问题: 存在"平时(生长过程中)表现好但终点不高"的植株, 静态(只看起点/终点)会漏掉。
      模型能否从早期数据 (GRM+TP1) out-of-sample 地选出这些植株?

真值表型 (用户算法, 纯实测, 非循环):
  每个时点 t: 该时点性状值前 top_frac(10%) 的个体记 1, 否则 0;
  个体得分 S(i) = (#优质时点) / (时点数)。
  变体: full = 全部时点; mid = 排除首尾时点 ("平时")。

评估链:
  ① 真值质量株 (得分前10%) 与 初始点/终点前10% 的重合度 → "静态盲区"是否存在;
  ② 各方法 (out-of-sample, fold-set 0, test=fold6) 选出前10%, 与真值质量株重合率+选择响应;
  ③ 方法对比 (同输入同协议): KNode-ODE vs KNode-ODE无GRM vs LSTM vs Transformer
     → 分离 "GRM/SNP 输入贡献" 与 "动力学模型族贡献";
  ④ 静态基线: 真TP1、真终点(oracle)、岭回归预测终点(现实静态GS)。

输出: outputs/reproduce/knode_staygreen_selection.json / .csv
用法: python knode_staygreen_selection.py [--seeds 42,1234] [--epochs 800]
"""
import json, math, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.001; WEIGHT_DECAY = 1e-4
OUT_DIR = Path('../outputs/reproduce')
DATA_DIR = Path('../data/maize_magic')
TOP_FRAC = 0.10

pheno = pd.read_csv(DATA_DIR / 'maize_trait_BLUP_50_NArmv_traits_afterClustering.csv')
cv_raw = pd.read_csv(DATA_DIR / 'cross_validation_folds_w_validation_set_maize.csv', index_col=0)
grm_full = pd.read_csv(DATA_DIR / 'GRM_330x330_methodA_from552.csv', index_col=0)

meta_cols = ['bio_ID', 'accession', 'DAS']
trait_cols = [c for c in pheno.columns if c not in meta_cols
              and not c.startswith('Unnamed') and c != '']
P = len(trait_cols)
times_all = np.array(sorted(pheno['DAS'].unique()), dtype=np.float64)
T = len(times_all)

common_ids = sorted(set(pheno['bio_ID'].astype(str)) & set(cv_raw.index))
N = len(common_ids)
acc = pheno.set_index('bio_ID')['accession'].to_dict()
grm_full.index = grm_full.index.astype(str); grm_full.columns = grm_full.columns.astype(str)
grm_raw = grm_full.loc[common_ids, common_ids].values.astype(np.float64)

Y_all = np.full((N, T, P), np.nan, dtype=np.float64)
for pi, trait in enumerate(trait_cols):
    sub = pheno[['bio_ID', 'DAS', trait]].copy(); sub['bio_ID'] = sub['bio_ID'].astype(str)
    Y_all[:, :, pi] = sub.pivot(index='bio_ID', columns='DAS', values=trait).loc[common_ids].values
assert not np.any(np.isnan(Y_all))

tn = (times_all - times_all.min()) / (times_all.max() - times_all.min() + 1e-8)
times_t = torch.tensor(tn, dtype=torch.float32, device=DEVICE)

pca = PCA(n_components=50)
S_all = pca.fit_transform(grm_raw).astype(np.float32)


# ---------------- 持续优质得分 (用户算法) ----------------
def sustained_score(Y, mid_only=False):
    """Y:(N,T,P) 真实单位 -> S:(N,P). 每时点 top_frac 记1, 得分=优质时点/时点数."""
    Nn, Tn, Pn = Y.shape
    lo, hi = (1, Tn - 1) if mid_only else (0, Tn)
    S = np.zeros((Nn, Pn))
    for t in range(lo, hi):
        for p in range(Pn):
            k = max(1, int(np.round(TOP_FRAC * Nn)))
            thr = np.sort(Y[:, t, p])[-k]
            S[:, p] += (Y[:, t, p] >= thr).astype(float)
    return S / max(hi - lo, 1)


def selection_metrics(score_pred, score_true):
    """同一评价集合内: top10% 重合率 + 选择响应."""
    Nn = len(score_true)
    k = max(1, int(np.round(TOP_FRAC * Nn)))
    tp = set(np.argsort(-score_true)[:k].tolist())
    pp = np.argsort(-score_pred)[:k]
    overlap = len(tp & set(pp.tolist())) / k
    resp = score_true[pp].mean(); resp_true = score_true[list(tp)].mean()
    rand = score_true.mean()
    return {'overlap': float(overlap), 'resp_pred': float(resp),
            'resp_true': float(resp_true), 'random': float(rand),
            'resp_pct': float((resp - rand) / (resp_true - rand + 1e-12))}


# ---------------- ODE 族 (rank-K Koopman, 默认 rank-1, 可选 GRM) ----------------
class ODEModel(nn.Module):
    """低秩 Koopman ODE: dh/dt = U (V^T h), U,V 为 P×K (rank=K), 由 cond 生成.
    rank=1 时退化为 u·(v·h) (原 KNode). rank>=2 提供 K 个指数模态 (镜像 arab 模块)."""
    def __init__(self, P=50, COND_IN=100, RANK=1):
        super().__init__()
        self.rank = max(int(RANK), 1)
        self.fc_U = nn.Linear(COND_IN, P * self.rank)
        self.fc_V = nn.Linear(COND_IN, P * self.rank)
        self._u = self._v = None

    def set_z(self, cond):
        self._u = self.fc_U(cond).view(-1, self.rank, self.fc_U.out_features // self.rank)
        self._v = self.fc_V(cond).view(-1, self.rank, self.fc_V.out_features // self.rank)

    def forward(self, t, h):
        dot = torch.einsum('bp,brp->br', h, self._v)
        return torch.einsum('br,brp->bp', dot, self._u)


class LSTMModel(nn.Module):
    """条件轨迹 LSTM: cond -> 初始隐态, 按时间步生成整条轨迹."""
    def __init__(self, P=50, COND_IN=100, HID=128):
        super().__init__()
        self.fc_h = nn.Linear(COND_IN, HID); self.fc_c = nn.Linear(COND_IN, HID)
        self.cell = nn.LSTMCell(1, HID); self.head = nn.Linear(HID, P)

    def forward(self, cond, ts):
        B = cond.shape[0]
        h = torch.tanh(self.fc_h(cond)); c = torch.tanh(self.fc_c(cond))
        outs = []
        for t in ts:
            h, c = self.cell(t.view(1).expand(B, 1), (h, c))
            outs.append(self.head(h))
        return torch.stack(outs, dim=1)


class TrajTransformer(nn.Module):
    """条件轨迹 Transformer: 时间token + 广播 cond -> 逐时点预测."""
    def __init__(self, P=50, COND_IN=100, D=96, NH=2, FF=256, NL=2):
        super().__init__()
        self.time_emb = nn.Linear(1, D); self.cond_proj = nn.Linear(COND_IN, D)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D, NH, FF, batch_first=True), NL)
        self.head = nn.Linear(D, P)

    def forward(self, cond, ts):
        B = cond.shape[0]
        te = self.time_emb(ts.view(-1, 1)).unsqueeze(0)     # (1,T,D)
        c = self.cond_proj(cond).unsqueeze(1)               # (B,1,D)
        out = self.enc(te + c)
        return self.head(out)                                # (B,T,P)


def build_cond(grm_pca, y1, cond_mode='full'):
    """cond 构造, cond_mode ∈ {full, tp1, grm, pad}:
      - full: [GRM_PCA, TP1]     (两者都在)
      - tp1 : [TP1]              (去掉 GRM)
      - grm : [GRM_PCA]          (去掉 TP1)
      - pad : [GRM_PCA, 0·TP1]   (TP1 置零, 维度与 full 相同)
    向后兼容: True -> full, False -> tp1 (旧 use_grm 调用)."""
    if cond_mode is True or cond_mode == 'full':
        return torch.cat([grm_pca, y1], dim=-1)
    if cond_mode is False or cond_mode == 'tp1':
        return y1
    if cond_mode == 'grm':
        return grm_pca
    if cond_mode == 'pad':
        return torch.cat([grm_pca, torch.zeros_like(y1)], dim=-1)
    raise ValueError(f'unknown cond_mode: {cond_mode!r}')


def cond_dim(grm_n, Pn, cond_mode='full'):
    """各 cond_mode 下模型 COND_IN 维度 (与 build_cond 输出一致)."""
    if cond_mode in ('full', 'pad'):
        return int(grm_n) + int(Pn)
    if cond_mode == 'grm':
        return int(grm_n)
    if cond_mode == 'tp1':
        return int(Pn)
    raise ValueError(f'unknown cond_mode: {cond_mode!r}')


def train_model(family, Ys_tr, grm_tr, y1_tr, seed, n_epochs, use_grm=True):
    torch.manual_seed(seed); np.random.seed(seed)
    B, Tn, Pn = Ys_tr.shape
    cond_in = (grm_tr.shape[1] + Pn) if use_grm else Pn
    if family == 'ode':
        m = ODEModel(Pn, cond_in).to(DEVICE)
    elif family == 'lstm':
        m = LSTMModel(Pn, cond_in).to(DEVICE)
    elif family == 'transformer':
        m = TrajTransformer(Pn, cond_in).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    y_tr = torch.tensor(Ys_tr.astype(np.float32), device=DEVICE)
    g_tr = torch.tensor(grm_tr, device=DEVICE)
    y1_tr_t = torch.tensor(y1_tr.astype(np.float32), device=DEVICE)
    best = None; best_loss = float('inf')
    for ep in range(n_epochs):
        m.train()
        cond = build_cond(g_tr, y1_tr_t, use_grm)
        if family == 'ode':
            m.set_z(cond)
            pred = euler_integrate(m, y1_tr_t, times_t)
        else:
            pred = m(cond, times_t)
        loss = nn.functional.mse_loss(pred, y_tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 50 == 0:
            with torch.no_grad():
                m.eval(); cond = build_cond(g_tr, y1_tr_t, use_grm)
                pv = euler_integrate(m, y1_tr_t, times_t) if family == 'ode' else m(cond, times_t)
                vl = nn.functional.mse_loss(pv[:, 1:], y_tr[:, 1:]).item()
            if not math.isnan(vl) and vl < best_loss:
                best_loss = vl
                best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    if best is None:
        best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best); m.eval()
    return m


def euler_integrate(model, h0, ts):
    B, D = h0.shape; T_len = len(ts)
    traj = torch.zeros(B, T_len, D, device=h0.device, dtype=h0.dtype); traj[:, 0] = h0
    h = h0
    for i in range(1, T_len):
        dt = max(ts[i].item() - ts[i - 1].item(), 1e-6)
        h = h + model(None, h) * dt; traj[:, i] = h
    return traj


def predict_real(m, family, grm, y1n, tr_min, tr_rng, cond_mode='full'):
    with torch.no_grad():
        g = torch.tensor(np.asarray(grm, dtype=np.float32), device=DEVICE)
        y = torch.tensor(np.asarray(y1n, dtype=np.float32), device=DEVICE)
        cond = build_cond(g, y, cond_mode)
        if family == 'ode':
            m.set_z(cond); pred = euler_integrate(m, y, times_t)
        else:
            pred = m(cond, times_t)
    return tr_min + tr_rng * pred.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,1234')
    ap.add_argument('--epochs', type=int, default=800)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(',') if x.strip()]
    print(f'Device: {DEVICE}  seeds={seeds}  N={N} T={T} P={P}  top_frac={TOP_FRAC}', flush=True)

    fc = cv_raw.loc[common_ids].iloc[:, 0].values.astype(int)
    tr = np.where((fc >= 1) & (fc <= 5))[0]
    te = np.where(fc == 6)[0]
    tr_min = np.nanmin(Y_all[tr], axis=(0, 1)); tr_max = np.nanmax(Y_all[tr], axis=(0, 1))
    tr_rng = np.maximum(tr_max - tr_min, 1e-8)
    Ys_tr = ((Y_all[tr] - tr_min) / tr_rng).astype(np.float32)
    y1_all = ((Y_all[:, 0, :] - tr_min) / tr_rng).astype(np.float32)
    S_tr = S_all[tr]; S_te = S_all[te]
    Y_te_real = Y_all[te]

    # ---------- 阶段①: 真值持续优质得分 (全面板, 描述性) + 静态盲区 ----------
    k_panel = max(1, int(np.round(TOP_FRAC * N)))
    res_true = {}
    for tag, mid in [('full', False), ('mid', True)]:
        S_true = sustained_score(Y_all, mid_only=mid)
        stat_tp1 = Y_all[:, 0, :]      # 初始点
        stat_fin = Y_all[:, -1, :]     # 终点
        ov_tp1 = []; ov_fin = []; miss_fin = []; miss_tp1 = []
        for p in range(P):
            q = set(np.argsort(-S_true[:, p])[:k_panel].tolist())
            t1 = set(np.argsort(-stat_tp1[:, p])[:k_panel].tolist())
            fn = set(np.argsort(-stat_fin[:, p])[:k_panel].tolist())
            ov_tp1.append(len(q & t1) / k_panel)
            ov_fin.append(len(q & fn) / k_panel)
            miss_fin.append(len(q - fn) / k_panel)
            miss_tp1.append(len(q - t1) / k_panel)
        res_true[tag] = {'overlap_with_tp1': float(np.mean(ov_tp1)),
                         'overlap_with_final': float(np.mean(ov_fin)),
                         'frac_missed_by_final': float(np.mean(miss_fin)),
                         'frac_missed_by_tp1': float(np.mean(miss_tp1))}
        print(f'\n[阶段① 真值持续优质 {tag} (top{TOP_FRAC:.0%}, 全面板 n={N})]')
        print(f'  与初始点top重合: {np.mean(ov_tp1):.3f}   与终点top重合: {np.mean(ov_fin):.3f}')
        print(f'  被终点漏掉(质量株但非终点top): {np.mean(miss_fin):.3f}   '
              f'被初始点漏掉: {np.mean(miss_tp1):.3f}', flush=True)

    # ---------- 阶段②③④: out-of-sample, 测试折对比 ----------
    S_true_te = sustained_score(Y_te_real, mid_only=True)   # 真值(测试折内相对, mid变体)
    scores_by_method = {}                                   # tag -> (n_te, P) 得分矩阵

    # 静态基线: 真TP1 / 真终点(oracle)
    scores_by_method['Static_TP1'] = Y_te_real[:, 0, :]
    scores_by_method['Static_final_oracle'] = Y_te_real[:, -1, :]

    # 静态岭回归: GRM+TP1 -> 终点值 (现实静态GS)
    X_tr = np.concatenate([S_tr, Ys_tr[:, 0, :]], axis=1)
    X_te = np.concatenate([S_te, y1_all[te]], axis=1)
    sp = np.zeros((len(te), P))
    for p in range(P):
        rg = Ridge(alpha=1.0).fit(X_tr, Ys_tr[:, -1, p])
        sp[:, p] = rg.predict(X_te)
    scores_by_method['Static_final_pred'] = sp

    # 动态模型: ODE(GRM+TP1), ODE(TP1 only), LSTM, Transformer —— 同一协议
    fams = [('ODE_GRM', 'ode', True), ('ODE_noGRM', 'ode', False),
            ('LSTM', 'lstm', True), ('Transformer', 'transformer', True)]
    for tag, fam, ug in fams:
        preds = np.zeros((len(te), T, P))
        for seed in seeds:
            m = train_model(fam, Ys_tr, S_tr, y1_all[tr], seed, a.epochs, use_grm=ug)
            preds += predict_real(m, fam, S_te, y1_all[te], tr_min, tr_rng, ug) / len(seeds)
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        scores_by_method[tag] = sustained_score(preds, mid_only=True)
        print(f'  [{tag}] 已训练并预测', flush=True)

    # ---- 统一计算: 重合率 / 选择响应 / 找回率 ----
    k_te = max(1, int(np.round(TOP_FRAC * len(te))))
    res = {}
    for tag, sc in scores_by_method.items():
        ov, rp = [], []
        rec_fin, rec_tp1 = [], []
        for p in range(P):
            m = selection_metrics(sc[:, p], S_true_te[:, p])
            ov.append(m['overlap']); rp.append(m['resp_pct'])
            q_top = set(np.argsort(-S_true_te[:, p])[:k_te].tolist())
            fin_top = set(np.argsort(-Y_te_real[:, -1, p])[:k_te].tolist())
            tp1_top = set(np.argsort(-Y_te_real[:, 0, p])[:k_te].tolist())
            mtop = set(np.argsort(-sc[:, p])[:k_te].tolist())
            miss_fin = q_top - fin_top; miss_tp1 = q_top - tp1_top
            if len(miss_fin) > 0: rec_fin.append(len(mtop & miss_fin) / len(miss_fin))
            if len(miss_tp1) > 0: rec_tp1.append(len(mtop & miss_tp1) / len(miss_tp1))
        res[tag] = {'overlap_mean': float(np.mean(ov)), 'resp_pct_mean': float(np.mean(rp)),
                    'recovery_missed_by_final': float(np.mean(rec_fin)) if rec_fin else float('nan'),
                    'recovery_missed_by_tp1': float(np.mean(rec_tp1)) if rec_tp1 else float('nan')}
        print(f'  [{tag}] overlap@10%={np.mean(ov):.3f} 响应%={np.mean(rp):.1f}  '
              f'找回被终点漏掉={np.mean(rec_fin):.3f}', flush=True)

    print('\n[对比表: out-of-sample 测试折, 真值=持续优质(mid, 排除首尾)]')
    print(f'  {"方法":20s} {"重合@10%":>9s} {"响应%":>7s} {"找回终点漏":>10s} {"找回TP1漏":>10s}')
    for tag in res:
        r = res[tag]
        print(f'  {tag:20s} {r["overlap_mean"]:9.3f} {r["resp_pct_mean"]:6.1f} '
              f'{r["recovery_missed_by_final"]:10.3f} {r["recovery_missed_by_tp1"]:10.3f}')

    out = {'top_frac': TOP_FRAC, 'seeds': seeds, 'n_epochs': a.epochs,
           'protocol': 'fold-set 0, folds1-5 train=val, fold6 test (out-of-sample); '
                       'score = 每时点前10%记1, 得分=时点占比; mid=排除首尾时点',
           'stage1_true_full_panel': res_true,
           'stage2_comparison_test_fold': res,
           'note': 'ODE_GRM=KNode rank-1 (GRM+TP1); ODE_noGRM=仅TP1(分离GRM/SNP贡献); '
                   'LSTM/Transformer=同输入同协议其他动态模型(分离模型族贡献); '
                   'Static_*=静态基线(TP1/终点/岭回归预测终点)'}
    with open(OUT_DIR / 'knode_staygreen_selection.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f'\nSaved: knode_staygreen_selection.json')


if __name__ == '__main__':
    main()
