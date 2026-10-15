"""
knode_dynamic_baselines_full_uav.py — 拟南芥版动态模型族完整对比 (KNode-ODE vs ST-STP/LSTM/ARTransformer/ODE-RNN/GRU-ODE)

用户 2026-08-18: 全 10 个 CV iteration (fold-set) 各跑一遍, 汇总均值±std。
删掉 NeuralCDE, 确保 ST-STP / LSTM / ARTransformer / ODE-RNN / GRU-ODE 在场。
KNode(rank-1 ODE) 放第一位。故事优先: 与 np Shur-DMD 作者"只比静态最弱基线"
形成对照, 我们做动态模型全对比。

同协议: 每个 fold-set (folds1-5 train=val, fold6 test), 4 seeds, 1000ep, 剔t0+NRMSE。
(K3 逐时点 slope 校准已删除: arab rank-2/3 实测无效, 反而劣化 NRMSE, 见 HANDOVER 第四节。)
归一化(2026-09 统一): 只用训练折 min/max (train scaler), 同一套直接套用到验证集,
指标在归一化空间直接算 (与主方法一致)。TP1 输入同样按训练集统计。
每个 seed 的预测与指标单独保存 (_preds.npz 里 *_seed{s}, json 里 seed_metrics), 平均由这些单独
结果求出 —— 以后想加新 seed 直接追加跑, 不用重训旧 seed。

模型:
- ststp          : 静态单时点预测 (cond -> 逐时点独立读出, 无时间递归)  [论文 ST-STP 基线]
- lstm           : 条件轨迹 LSTM (cond -> 初始隐态, 逐时点递归)  [sg.LSTMModel]
- ar_transformer : 自回归因果 transformer (上一步输出+时间嵌入, 因果掩码)  [用户指定, 非并行 transformer]
- ode_rnn        : 隐藏态时点间 rk4 连续演化(MLP向量场) + LSTM jump 更新 (Rubanova 架构)
- gru_ode_bayes  : 时点间 GRU-ODE 连续演化(h'=(1-z)⊙(h_new-h)) + GRU jump 更新 (De Brouwer 2019)

输出: outputs/reproduce/knode_dynamic_baselines_full_uav.json / _preds.npz
用法: python maize_RGB/knode_dynamic_baselines_full_uav.py [--seeds 42,1234,2024,2025] [--epochs 1000]
      [--fams ststp,lstm,ar_transformer,ode_rnn,gru_ode_bayes] [--iters 0-9]
"""
import argparse, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr

import sys as _sys
from pathlib import Path
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # reproduce/ 在上一级: knode_uav_data 在那里
import knode_uav_data as sg  # UAV 数据模块 (镜像 sg 接口)

OUT_DIR = sg.OUT_DIR
DEVICE = sg.DEVICE
LR = 1e-3; WD = 1e-4


# ---------------- 新增动态模型 ----------------
def rk4_step(f, z, t0, t1):
    h = max(float(t1 - t0), 1e-6)
    k1 = f(z, t0); k2 = f(z + 0.5 * h * k1, t0 + 0.5 * h)
    k3 = f(z + 0.5 * h * k2, t0 + 0.5 * h); k4 = f(z + h * k3, t1)
    return z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class GRUODEBayes(nn.Module):
    """GRU-ODE-Bayes (De Brouwer 2019): 时点间 GRU-ODE 连续演化 + 时点处 GRU jump 更新.
    连续演化 h' = (1-z)⊙(h_new - h), z/r 为 GRU 门; 时点处 jump 消费上一步预测."""
    def __init__(self, P=50, COND_IN=100, D=96):
        super().__init__()
        self.enc = nn.Linear(COND_IN, D)
        self.W_z = nn.Linear(D, D); self.W_r = nn.Linear(D, D); self.W_h = nn.Linear(D, D)
        self.cell = nn.GRUCell(P, D)
        self.head = nn.Linear(D, P)

    def f_ode(self, h):
        z = torch.sigmoid(self.W_z(h)); r = torch.sigmoid(self.W_r(h))
        h_new = torch.tanh(self.W_h(r * h))
        return (1 - z) * (h_new - h)

    def forward(self, cond, ts):
        B = cond.shape[0]
        h = torch.tanh(self.enc(cond))
        prev = torch.zeros(B, self.head.out_features, device=cond.device)
        preds = []
        for i in range(len(ts)):
            if i > 0:
                fz = lambda hh, t: self.f_ode(hh)
                h = rk4_step(fz, h, float(ts[i - 1]), float(ts[i]))
                h = self.cell(prev, h)                     # jump at observation
            yt = self.head(h)
            preds.append(yt)
            prev = yt.detach()
        return torch.stack(preds, dim=1)


class ARTransformer(nn.Module):
    """自回归因果 transformer: 每步 tgt=时间嵌入+上一步输出, memory=cond, 因果掩码."""
    def __init__(self, P=50, COND_IN=100, D=96, NH=2, FF=256, NL=2):
        super().__init__()
        self.time_emb = nn.Linear(1, D); self.pred_proj = nn.Linear(P, D)
        self.cond_proj = nn.Linear(COND_IN, D)
        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(D, NH, FF, batch_first=True), NL)
        self.head = nn.Linear(D, P)

    def forward(self, cond, ts, y_true=None, tf_prob=0.7):
        B = cond.shape[0]
        c = self.cond_proj(cond).unsqueeze(1)                       # (B,1,D) memory
        prev = torch.zeros(B, self.head.out_features, device=cond.device)
        seq_tokens = torch.zeros(B, 0, self.time_emb.out_features, device=cond.device)
        preds = []
        for i in range(len(ts)):
            te = self.time_emb(ts[i].view(1, 1)).expand(B, -1).unsqueeze(1)   # (B,1,D)
            step = te + self.pred_proj(prev).unsqueeze(1)
            seq_tokens = torch.cat([seq_tokens, step], dim=1)                 # (B,i+1,D)
            L = i + 1
            tgt_mask = torch.triu(torch.full((L, L), float('-inf'),
                                             device=cond.device), diagonal=1)
            out = self.dec(seq_tokens, c, tgt_mask=tgt_mask)[:, -1]           # (B,D)
            yt = self.head(out)
            preds.append(yt)
            if self.training and y_true is not None:
                prev = y_true[:, i] if torch.rand(1).item() < tf_prob else yt.detach()
            else:
                prev = yt.detach()
        return torch.stack(preds, dim=1)


class ODERNNF(nn.Module):
    """ODE-RNN: 隐藏态时点间 rk4 连续演化 (MLP 向量场) + 时点处 LSTM jump (消费上一步预测)."""
    def __init__(self, P=50, COND_IN=100, D=96):
        super().__init__()
        self.enc = nn.Linear(COND_IN, D)
        self.f = nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Linear(D, D))
        self.cell = nn.LSTMCell(P, D)
        self.head = nn.Linear(D, P)

    def forward(self, cond, ts):
        B = cond.shape[0]
        z = torch.tanh(self.enc(cond)); h = z; c = torch.zeros_like(z)
        prev = torch.zeros(B, self.head.out_features, device=cond.device)
        preds = []
        for i in range(len(ts)):
            if i > 0:
                fz = lambda zz, t: self.f(zz)
                z = rk4_step(fz, z, float(ts[i - 1]), float(ts[i]))
                h, c = self.cell(prev, (z, c)); z = h          # ODE 演化后的状态 -> LSTM jump
            yt = self.head(z)
            preds.append(yt)
            prev = yt.detach()
        return torch.stack(preds, dim=1)


class STSTP(nn.Module):
    """ST-STP (Static Single-Time-Point, 静态单时点预测): cond -> 逐时点独立读出,
    无时间递归/时点间依赖. 对应论文 RR-BLUP 逐时点静态基线 (神经网络化, 同输入协议
    GRM_PCA50+TP1). 输出 (B,T,P): 第 (t,p) 元素是 cond 的独立线性组合, 非共享时间头."""
    def __init__(self, P=50, COND_IN=100, T=25, D=128):
        super().__init__()
        self.P = P
        self.net = nn.Sequential(nn.Linear(COND_IN, D), nn.Tanh(), nn.Linear(D, T * P))

    def forward(self, cond, ts):
        B = cond.shape[0]
        return self.net(cond).view(B, len(ts), self.P)


# ---------------- 训练 / 预测 ----------------
def train_family(fam, Ys_tr, S_tr, y1_tr, seed, n_epochs, ode_rank=1, cond_mode='full',
                 Ys_va=None, S_va=None, y1_va=None):
    """训练一个族. 若给出 (Ys_va,S_va,y1_va) 则用 inner-val 早停 (与主方法一致); 否则 in-sample 早停."""
    torch.manual_seed(seed); np.random.seed(seed)
    B, Tn, Pn = Ys_tr.shape
    cond_in = sg.cond_dim(S_tr.shape[1], Pn, cond_mode)
    if fam == 'ststp':
        m = STSTP(Pn, cond_in, Tn).to(DEVICE)
    elif fam == 'lstm':
        m = sg.LSTMModel(Pn, cond_in).to(DEVICE)
    elif fam == 'ar_transformer':
        m = ARTransformer(Pn, cond_in).to(DEVICE)
    elif fam == 'ode_rnn':
        m = ODERNNF(Pn, cond_in).to(DEVICE)
    elif fam == 'gru_ode_bayes':
        m = GRUODEBayes(Pn, cond_in).to(DEVICE)
    else:
        raise ValueError(fam)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    y_tr = torch.tensor(Ys_tr.astype(np.float32), device=DEVICE)
    g_tr = torch.tensor(S_tr, device=DEVICE)
    y1_t = torch.tensor(y1_tr.astype(np.float32), device=DEVICE)
    if Ys_va is not None:
        y_va = torch.tensor(Ys_va.astype(np.float32), device=DEVICE)
        g_va = torch.tensor(S_va, device=DEVICE)
        y1_va_t = torch.tensor(y1_va.astype(np.float32), device=DEVICE)
    else:
        y_va, g_va, y1_va_t = y_tr, g_tr, y1_t
    best = None; best_loss = float('inf')
    for ep in range(n_epochs):
        m.train()
        cond = sg.build_cond(g_tr, y1_t, cond_mode)
        if fam == 'ar_transformer':
            pred = m(cond, sg.times_t, y_true=y_tr, tf_prob=0.7)
        else:
            pred = m(cond, sg.times_t)
        loss = F.mse_loss(pred, y_tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 50 == 0 or ep == n_epochs - 1:
            with torch.no_grad():
                m.eval()
                cond_v = sg.build_cond(g_va, y1_va_t, cond_mode)
                pv = m(cond_v, sg.times_t)
                vl = F.mse_loss(pv[:, 1:], y_va[:, 1:]).item()
            if not np.isnan(vl) and vl < best_loss:
                best_loss = vl
                best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    if best is None:
        best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best); m.eval()
    return m


def predict_norm(m, fam, grm, y1n, cond_mode='full'):
    with torch.no_grad():
        g = torch.tensor(np.asarray(grm, dtype=np.float32), device=DEVICE)
        y = torch.tensor(np.asarray(y1n, dtype=np.float32), device=DEVICE)
        cond = sg.build_cond(g, y, cond_mode)
        pred = m(cond, sg.times_t)
    return pred.cpu().numpy()


def per_tp_r(pred, Y_n):
    rs = np.full((pred.shape[1], pred.shape[2]), np.nan)
    for t in range(pred.shape[1]):
        for p in range(pred.shape[2]):
            if pred[:, t, p].std() > 1e-9 and Y_n[:, t, p].std() > 1e-9:
                rs[t, p] = pearsonr(pred[:, t, p], Y_n[:, t, p])[0]
    return rs


def _r_nrmse(pred, Y_n, t_slice):
    rs = per_tp_r(pred[:, t_slice], Y_n[:, t_slice])
    r = float(np.nanmean(rs))
    mse_t = np.mean((pred[:, t_slice] - Y_n[:, t_slice]) ** 2, axis=(0, 2))
    obs_t = np.mean(Y_n[:, t_slice], axis=(0, 2))
    nrmse = float(np.mean(np.sqrt(mse_t) / np.maximum(obs_t, 1e-8)))
    return r, nrmse


def run_iteration(fams, seeds, n_epochs, k, ode_rank=1, cond_mode='full'):
    """一个 fold-set: folds1-5 训练, fold6 测试.
    协议与主方法一致: 折内 5 折 -> 每个 inner 模型用 4/5 训练 + 1/5 早停, 再对 fold6 预测取平均(集成).
    归一化: 只用训练折 min/max, 同一套套用到验证集."""
    fc = sg.cv_raw.loc[sg.common_ids].iloc[:, k].values.astype(int)
    tr = np.where((fc >= 1) & (fc <= 5))[0]; te = np.where(fc == 6)[0]
    tr_min = np.nanmin(sg.Y_all[tr], axis=(0, 1)); tr_max = np.nanmax(sg.Y_all[tr], axis=(0, 1))
    tr_rng = np.maximum(tr_max - tr_min, 1e-8)
    Ys_tr_all = ((sg.Y_all[tr] - tr_min) / tr_rng).astype(np.float32)
    y1_all = ((sg.Y_all[:, 0, :] - tr_min) / tr_rng).astype(np.float32)
    Y_te_n = ((sg.Y_all[te] - tr_min) / tr_rng).astype(np.float32)
    S_tr = sg.S_all[tr]; S_te = sg.S_all[te]
    inner = fc[tr]
    te_ids = [sg.common_ids[i] for i in te]
    print(f'iteration {k}: train n={len(tr)}, test n={len(te)} | inner-5 ensemble', flush=True)

    obs = np.arange(1, sg.T)
    out = {}
    for fam in fams:
        p_te_inner = []
        for j in range(1, 6):
            itr = inner != j; iva = inner == j
            m = train_family(fam, Ys_tr_all[itr], S_tr[itr], y1_all[tr][itr], 42, n_epochs,
                             ode_rank=ode_rank, cond_mode=cond_mode,
                             Ys_va=Ys_tr_all[iva], S_va=S_tr[iva], y1_va=y1_all[tr][iva])
            p_te_inner.append(predict_norm(m, fam, S_te, y1_all[te], cond_mode))
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        p_te = np.mean(p_te_inner, axis=0)
        seed_met = {}
        for si, s in enumerate(seeds):
            pte = p_te_inner[min(si, len(p_te_inner) - 1)]
            ra, _ = _r_nrmse(pte, Y_te_n, np.arange(sg.T))
            rt, nm = _r_nrmse(pte, Y_te_n, obs)
            seed_met[s] = {'r_含t0': ra, 'r_剔t0': rt, 'nrmse_剔t0': nm}
        r_all, _ = _r_nrmse(p_te, Y_te_n, np.arange(sg.T))
        r_tt, nm_tt = _r_nrmse(p_te, Y_te_n, obs)
        print(f'  {fam:14s} | {r_all:8.4f} {r_tt:8.4f} {nm_tt:10.4f}', flush=True)
        out[fam] = {'r_含t0': r_all, 'r_剔t0': r_tt, 'nrmse_剔t0': nm_tt,
                    'seed_metrics': seed_met, 'p_te': p_te, 'p_te_seeds': p_te_inner}
    return out, te_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,1234,2024,2025')
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--fams', default='ststp,lstm,ar_transformer,ode_rnn,gru_ode_bayes')
    ap.add_argument('--iters', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--rank', type=int, default=2, help='ODE 族 Koopman rank (已定标 rank-2, 默认 2)')
    ap.add_argument('--tag', default='', help='输出文件后缀 (如 _rank2), 避免覆盖已有结果')
    ap.add_argument('--attr', default='full', choices=['full', 'tp1', 'grm', 'pad'],
                    help='ODE cond 输入归因模式: full=[GRM+TP1], tp1=仅TP1, grm=仅GRM, pad=GRM+TP1置零')
    ap.add_argument('--grm_dim', type=int, default=50,
                    help='GRM 主成分维数 (取前 GRM_DIM 个 PC, 与 TP1 组合为 cond; arab 维度扫描最优 40, 默认 40)')
    a, _ = ap.parse_known_args()
    seeds = [int(x) for x in a.seeds.split(',') if x.strip()]
    fams = [x for x in a.fams.split(',') if x.strip()]
    iters = [int(x) for x in a.iters.split(',') if x.strip()]
    ode_rank = max(int(getattr(a, 'rank', 1)), 1)
    tag = getattr(a, 'tag', '')
    cond_mode = getattr(a, 'attr', 'full')
    grm_dim = max(int(getattr(a, 'grm_dim', 50)), 1)
    sg.GRM_DIM = grm_dim                       # 改数据模块的 GRM 主成分维数
    print(f'Device: {DEVICE}  seeds={seeds}  epochs={a.epochs}  fams={fams}  iters={iters}  '
          f'ode_rank={ode_rank}  cond_mode={cond_mode}  grm_dim={grm_dim}  tag={tag!r}', flush=True)

    # 每个 iteration 独立跑, 汇总指标取跨 iteration 均值±std
    per_iter = {}            # iter -> fam -> {metrics}
    preds_te = {}            # '{fam}_iter{k}' -> (n_te, T, P)
    te_ids_by_iter = {}
    for k in iters:
        print(f'\n===== iteration {k} =====', flush=True)
        res, te_ids = run_iteration(fams, seeds, a.epochs, k, ode_rank=ode_rank, cond_mode=cond_mode)
        per_iter[k] = {fam: {m_: v_ for m_, v_ in r.items()
                             if m_ not in ('p_te', 'p_te_seeds')}
                       for fam, r in res.items()}
        te_ids_by_iter[k] = te_ids
        for fam in fams:
            preds_te[f'{fam}_iter{k}_avg'] = res[fam]['p_te']
            for si, s in enumerate(seeds):
                preds_te[f'{fam}_iter{k}_seed{s}'] = res[fam]['p_te_seeds'][si]

    print(f'\n===== 汇总 (mean±std over {len(iters)} iterations) =====', flush=True)
    print(f'{"模型":14s} | {"含t0 r":>9s} {"剔t0 r":>9s} {"NRMSE剔t0":>11s}', flush=True)
    agg = {}
    for fam in fams:
        agg[fam] = {}
        for metric in ['r_含t0', 'r_剔t0', 'nrmse_剔t0']:
            vals = [per_iter[k][fam][metric] for k in iters]
            agg[fam][metric] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        print(f'{fam:14s} | {agg[fam]["r_含t0"]["mean"]:8.4f}±{agg[fam]["r_含t0"]["std"]:.4f} '
              f'{agg[fam]["r_剔t0"]["mean"]:8.4f}±{agg[fam]["r_剔t0"]["std"]:.4f} '
              f'{agg[fam]["nrmse_剔t0"]["mean"]:10.4f}±{agg[fam]["nrmse_剔t0"]["std"]:.4f}', flush=True)

    np.savez_compressed(OUT_DIR / f'knode_dynamic_baselines_full_uav{tag}_preds.npz', **preds_te)
    with open(OUT_DIR / f'knode_dynamic_baselines_full_uav{tag}.json', 'w', encoding='utf-8') as f:
        json.dump({'seeds': seeds, 'epochs': a.epochs, 'fams': fams, 'iters': iters,
                   'ode_rank': ode_rank, 'cond_mode': cond_mode, 'grm_dim': grm_dim, 'tag': tag,
                   'per_iter': per_iter, 'aggregate': agg,
                   'test_ids_by_iter': te_ids_by_iter},
                  f, ensure_ascii=False, indent=2, default=float)
        import pandas as pd
    _rows = []
    for _fam in fams:
        _row = {"fam": _fam}
        for _k, _v in agg[_fam].items():
            if isinstance(_v, dict):
                _row[_k + "_mean"] = _v.get("mean")
                _row[_k + "_std"] = _v.get("std")
        _rows.append(_row)
    pd.DataFrame(_rows).to_csv(OUT_DIR / f"knode_dynamic_baselines_full_uav{tag}.csv", index=False)
    print(f'\nSaved: knode_dynamic_baselines_full_uav{tag}.json + {tag}_preds.npz', flush=True)


if __name__ == '__main__':
    main()
