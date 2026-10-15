"""
knode_operator_ablation_arab.py — 算子价值消融 (拟南芥, fold-set 0, 单 iteration)

审稿人攻击面:
  1. hyper  : 逐基因型超网络生成 (u,v), 经 ODE 积分  (参考, cond_variants d40 剔t0=0.4722)
  2. direct : 不用算子/ODE, cond->MLP->直接回归整条轨迹 y(B,T,P)  ← 算子是否多此一举
  3. shared : u,v 跨基因型共享 (单个可学习参数对), 只 h0=TP1 变     ← 逐基因型算子是否必要
             (直接检验 MegaLMM "共享动力学" 前提)
三者 cond 输入 [GRM_PC, TP1] 一致, h0=TP1 锚点一致, 唯一差异是机制.
协议: 与 knode_cond_variants_arab.py 逐行一致 (fold-set 0 / 4 seeds / 1000ep / rank=2 / grm_dim=40).
输出: outputs/reproduce/knode_operator_ablation_arab.json
"""
import argparse, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr

import knode_arab_data as sg

OUT_DIR = sg.OUT_DIR
DEVICE = sg.DEVICE
LR = 1e-3; WD = 1e-4


class OperatorAblation(nn.Module):
    """统一容器: hyper/direct/shared 三机制, 前向接口分两路 (ODE forward 或 direct_forward)."""
    def __init__(self, P, GRM_N, TP_N, RANK=2, HID=128, variant='hyper', T=14):
        super().__init__()
        self.variant = variant
        self.rank = RANK
        cond_dim = int(GRM_N) + int(TP_N)
        if variant == 'hyper':
            self.enc = nn.Sequential(nn.Linear(cond_dim, HID), nn.Tanh())
            self.hyper_U = nn.Linear(HID, P * RANK)
            self.hyper_V = nn.Linear(HID, P * RANK)
        elif variant == 'shared':
            # u,v 为共享可学习参数, 跨基因型同一份 (只 h0 变). 与 hyper 同 rank/P 布局.
            self.U = nn.Parameter(0.1 * torch.randn(RANK, P))
            self.V = nn.Parameter(0.1 * torch.randn(RANK, P))
        elif variant == 'direct':
            self.enc = nn.Sequential(nn.Linear(cond_dim, HID), nn.Tanh())
            self.head = nn.Linear(HID, P * T)     # 直接回归整条轨迹
        else:
            raise ValueError(variant)
        self._u = self._v = None

    def set_z(self, grm, tp1):
        B = grm.shape[0]
        if self.variant == 'hyper':
            cond = torch.cat([grm, tp1], dim=-1)
            h = self.enc(cond)
            self._u = self.hyper_U(h).view(B, self.rank, -1)
            self._v = self.hyper_V(h).view(B, self.rank, -1)
        elif self.variant == 'shared':
            self._u = self.U.unsqueeze(0).expand(B, -1, -1)
            self._v = self.V.unsqueeze(0).expand(B, -1, -1)

    def forward(self, t, h):   # ODE 积分调用 (hyper/shared): dh/dt = U(V^T h)
        dot = torch.einsum('bp,brp->br', h, self._v)
        return torch.einsum('br,brp->bp', dot, self._u)

    def direct_forward(self, grm, tp1):   # direct 调用: 直接回归整条轨迹
        cond = torch.cat([grm, tp1], dim=-1)
        return self.head(self.enc(cond))


# ---------------- 训练 / 预测 / 指标 (同 cond_variants 协议) ----------------
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


def _predict(m, grm, y1n):
    with torch.no_grad():
        g = torch.tensor(np.asarray(grm, dtype=np.float32), device=DEVICE)
        y = torch.tensor(np.asarray(y1n, dtype=np.float32), device=DEVICE)
        B = g.shape[0]
        if m.variant == 'direct':
            pred = m.direct_forward(g, y).view(B, len(sg.times_t), -1)
        else:
            m.set_z(g, y)
            pred = sg.euler_integrate(m, y, sg.times_t)
    return pred.cpu().numpy()


def train_one(variant, Ys_tr, S_tr, y1_tr, seed, n_epochs, ode_rank, grm_dim):
    torch.manual_seed(seed); np.random.seed(seed)
    B, Tn, Pn = Ys_tr.shape
    m = OperatorAblation(Pn, grm_dim, Pn, ode_rank, variant=variant, T=Tn).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    y_tr = torch.tensor(Ys_tr.astype(np.float32), device=DEVICE)
    g_tr = torch.tensor(S_tr, device=DEVICE)
    y1_t = torch.tensor(y1_tr.astype(np.float32), device=DEVICE)
    best = None; best_loss = float('inf')
    for ep in range(n_epochs):
        m.train()
        if variant == 'direct':
            pred = m.direct_forward(g_tr, y1_t).view(B, Tn, Pn)
        else:
            m.set_z(g_tr, y1_t)
            pred = sg.euler_integrate(m, y1_t, sg.times_t)
        loss = F.mse_loss(pred, y_tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 50 == 0:
            with torch.no_grad():
                m.eval()
                if variant == 'direct':
                    pv = m.direct_forward(g_tr, y1_t).view(B, Tn, Pn)
                else:
                    m.set_z(g_tr, y1_t)
                    pv = sg.euler_integrate(m, y1_t, sg.times_t)
                vl = F.mse_loss(pv[:, 1:], y_tr[:, 1:]).item()
            if not np.isnan(vl) and vl < best_loss:
                best_loss = vl
                best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    if best is None:
        best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best); m.eval()
    return m


def run_iteration(variant, seeds, n_epochs, k, ode_rank, grm_dim):
    fc = sg.cv_raw.loc[sg.common_ids].iloc[:, k].values.astype(int)
    tr = np.where((fc >= 1) & (fc <= 5))[0]; te = np.where(fc == 6)[0]
    tr_min = np.nanmin(sg.Y_all[tr], axis=(0, 1)); tr_max = np.nanmax(sg.Y_all[tr], axis=(0, 1))
    tr_rng = np.maximum(tr_max - tr_min, 1e-8)
    te_min = np.nanmin(sg.Y_all[te], axis=(0, 1)); te_max = np.nanmax(sg.Y_all[te], axis=(0, 1))
    te_rng = np.maximum(te_max - te_min, 1e-8)
    Ys_tr = ((sg.Y_all[tr] - tr_min) / tr_rng).astype(np.float32)
    y1_all = ((sg.Y_all[:, 0, :] - tr_min) / tr_rng).astype(np.float32)
    Y_te_n = ((sg.Y_all[te] - te_min) / te_rng).astype(np.float32)
    S_tr = sg.get_S(grm_dim)[tr]; S_te = sg.get_S(grm_dim)[te]
    obs = np.arange(1, sg.T)
    print(f'  iteration {k}: train n={len(tr)}, test n={len(te)}', flush=True)
    p_te_s = []
    for seed in seeds:
        m = train_one(variant, Ys_tr, S_tr, y1_all[tr], seed, n_epochs, ode_rank, grm_dim)
        p_te_s.append(_predict(m, S_te, y1_all[te]))
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    p_te_s = [((p * tr_rng + tr_min - te_min) / te_rng).astype(np.float32) for p in p_te_s]
    p_te = np.mean(p_te_s, axis=0)
    r_all, _ = _r_nrmse(p_te, Y_te_n, np.arange(sg.T))
    r_tt, nm_tt = _r_nrmse(p_te, Y_te_n, obs)
    # 逐时点 PCC (剔除 t0), 看 direct 在哪些时点掉
    rs_all = per_tp_r(p_te, Y_te_n)
    per_tp = [round(float(np.nanmean(rs_all[t, :])), 4) for t in range(1, sg.T)]
    per_seed = []
    for p_s in p_te_s:
        r_s, _ = _r_nrmse(p_s, Y_te_n, obs)
        per_seed.append(r_s)
    return {'r_含t0': r_all, 'r_剔t0': r_tt, 'nrmse_剔t0': nm_tt,
            'per_tp_剔t0': per_tp, 'per_seed_剔t0': [round(x, 4) for x in per_seed]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default='hyper,direct,shared',
                    help='逗号分隔, 至少一个 ∈ {hyper, direct, shared}')
    ap.add_argument('--seeds', default='42,1234,2024,2025')
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--iters', default='0')
    ap.add_argument('--rank', type=int, default=2)
    ap.add_argument('--grm_dim', type=int, default=40)
    a = ap.parse_args()
    variants = [v.strip() for v in a.variants.split(',') if v.strip()]
    seeds = [int(x) for x in a.seeds.split(',') if x.strip()]
    iters = [int(x) for x in a.iters.split(',') if x.strip()]
    sg.GRM_DIM = a.grm_dim
    print(f'Device:{DEVICE} variants={variants} seeds={seeds} epochs={a.epochs} '
          f'iters={iters} rank={a.rank} grm_dim={a.grm_dim}', flush=True)

    out = {'variants': variants, 'seeds': seeds, 'epochs': a.epochs, 'iters': iters,
           'rank': a.rank, 'grm_dim': a.grm_dim, 'per_variant': {}}
    for variant in variants:
        per_iter = {}
        for k in iters:
            print(f'===== iteration {k} variant={variant} =====', flush=True)
            per_iter[k] = run_iteration(variant, seeds, a.epochs, k, a.rank, a.grm_dim)
            pi = per_iter[k]
            print(f'  {variant:7s} | 含t0 r={pi["r_含t0"]:.4f} 剔t0 r={pi["r_剔t0"]:.4f} '
                  f'NRMSE={pi["nrmse_剔t0"]:.4f} | per_seed={" ".join(f"{x:.4f}" for x in pi["per_seed_剔t0"])}', flush=True)
            print(f'      per_tp: {pi["per_tp_剔t0"]}', flush=True)
        out['per_variant'][variant] = per_iter

    with open(OUT_DIR / 'knode_operator_ablation_arab.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print('\nSaved: knode_operator_ablation_arab.json', flush=True)


if __name__ == '__main__':
    main()
