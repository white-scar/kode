"""
knode_cond_variants_arab.py — cond->(u,v) 瓶颈方向探针 (拟南芥, 默认单 iteration)

方向一 (修 cond->(u,v) 瓶颈, 不换输入):
  baseline  对照: 原 ODEModel, cond=[GRM_PC,TP1] -> 两个 Linear 生成 u,v
  hyper  (1a) 逐基因型超网络直接生成 (u,v): cond -> MLP(128) -> u,v  (非线性算子映射)
  attn   (1a') 注意力头融合超网络 (transformer 启发①, 用户 2026-08-18): hyper_enc 输出 h 作 query,
              softmax 注意力于 M=64 个可学习算符基向量 (K/V), 注意力加权和生成 u,v.
              = 学习到的 data-dependent 跨基因型核 (注意力) 直入算子生成, 保留显式算子/OOF 安全.
  kernel (1b) 遗传距离->动态相似度核平滑: GRM-PC 按高斯核加权平均后进 cond
             (遗传近的基因型 -> 近似的 GRM 输入, 归纳偏置直接作用)
  scale  (1c) 遗传输入专门调制尺度: u 由 TP1 生成, v 共享(幅度非时间常数, L3~L2),
             GRM 只调制每基因型幅度增益 s=1+tanh(...)
方向二 (可学习遗传嵌入, 替换冻结 PCA):
  genemb      top-PC -> 可学习 MLP (残差+零初始化, 起于恒等) -> 遗传嵌入, 端到端

协议与 knode_dynamic_baselines_full_arab.py 完全一致 (同归一化/同 fold-set 划分/4 seeds/1000ep),
只跑 ode 族 + 指定 variant. --iters 默认 0 (单 iteration 探针).
"""
import argparse, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr

import knode_arab_data as sg

OUT_DIR = sg.OUT_DIR
DEVICE = sg.DEVICE
LR = 1e-3; WD = 1e-4


class CondVariantODE(nn.Module):
    """低秩 Koopman ODE (rank K), cond->(u,v) 变体. dh/dt = U(V^T h), U,V: B×K×P.
    forward(t,h) 兼容 sg.euler_integrate (它调用 model(None,h))."""
    def __init__(self, P=45, GRM_N=40, TP_N=45, RANK=2, variant='baseline', HID=128):
        super().__init__()
        self.rank = max(int(RANK), 1)
        self.variant = variant
        cond_dim = int(GRM_N) + int(TP_N)
        self.fc_U = nn.Linear(cond_dim, P * self.rank)      # baseline / kernel / genemb
        self.fc_V = nn.Linear(cond_dim, P * self.rank)
        if variant in ('hyper', 'attn'):
            self.hyper_enc = nn.Sequential(nn.Linear(cond_dim, HID), nn.Tanh())
        if variant == 'hyper':
            self.hyper_U = nn.Linear(HID, P * self.rank)
            self.hyper_V = nn.Linear(HID, P * self.rank)
        if variant == 'attn':
            self.hyper_U = nn.Linear(HID, P * self.rank)       # 主路径: hyper (保基因型特异性)
            self.hyper_V = nn.Linear(HID, P * self.rank)
            self.attn_M = 64
            self.attn_q = nn.Linear(HID, 64)                          # 超网络隐态 -> 查询
            self.attn_K = nn.Parameter(1.0 * torch.randn(64, 64))     # logits 有区分度
            self.attn_VU = nn.Parameter(torch.zeros(64, P * self.rank))  # 零初始化残差
            self.attn_VV = nn.Parameter(torch.zeros(64, P * self.rank))
        if variant == 'scale':
            self.fc_U_tp = nn.Linear(TP_N, P * self.rank)   # 形状从 TP1
            self.V_shared = nn.Parameter(0.1 * torch.randn(self.rank, P))   # (rank, P), 与 _v 布局一致
            self.fc_scale = nn.Sequential(nn.Linear(GRM_N, 16), nn.Tanh(), nn.Linear(16, 1))
        if variant == 'genemb':
            self.gen_emb = nn.Sequential(nn.Linear(GRM_N, 64), nn.Tanh(), nn.Linear(64, GRM_N))
            nn.init.zeros_(self.gen_emb[2].weight)          # 残差 => 起于恒等 (baseline)
            nn.init.zeros_(self.gen_emb[2].bias)
        self._u = self._v = None

    def _smooth(self, S):
        """高斯核加权平均: 遗传近 (GRM-PC 距离近) 的基因型共享近似的遗传输入."""
        d2 = torch.cdist(S, S) ** 2
        with torch.no_grad():
            tau = torch.median(d2) * 0.5 + 1e-6
        K = torch.exp(-d2 / tau)
        K = K / (K.sum(dim=-1, keepdim=True) + 1e-6)
        return K @ S

    def set_z(self, grm, tp1):
        B = grm.shape[0]; v = self.variant
        if v == 'baseline':
            cond = torch.cat([grm, tp1], dim=-1)
            u = self.fc_U(cond).view(B, self.rank, -1)
            vv = self.fc_V(cond).view(B, self.rank, -1)
        elif v == 'hyper':
            cond = torch.cat([grm, tp1], dim=-1)
            h = self.hyper_enc(cond)
            u = self.hyper_U(h).view(B, self.rank, -1)
            vv = self.hyper_V(h).view(B, self.rank, -1)
        elif v == 'attn':
            cond = torch.cat([grm, tp1], dim=-1)
            h = self.hyper_enc(cond)
            q = self.attn_q(h)                                        # (B, 64)
            a = torch.softmax(q @ self.attn_K.T / 8.0, dim=-1)        # /sqrt(64)
            # 残差: hyper 主路径 + 零初始化注意力修正 (起于 hyper, 只往好里学)
            u = (self.hyper_U(h) + (a @ self.attn_VU)).view(B, self.rank, -1)
            vv = (self.hyper_V(h) + (a @ self.attn_VV)).view(B, self.rank, -1)
        elif v == 'kernel':
            g = self._smooth(grm)
            cond = torch.cat([g, tp1], dim=-1)
            u = self.fc_U(cond).view(B, self.rank, -1)
            vv = self.fc_V(cond).view(B, self.rank, -1)
        elif v == 'scale':
            u = self.fc_U_tp(tp1).view(B, self.rank, -1)
            s = (1.0 + torch.tanh(self.fc_scale(grm))).view(B, 1, 1)   # 幅度增益 ∈ (0,2)
            u = u * s
            vv = self.V_shared.unsqueeze(0).expand(B, -1, -1)
        elif v == 'genemb':
            g = grm + self.gen_emb(grm)                                # 残差, 起于恒等
            cond = torch.cat([g, tp1], dim=-1)
            u = self.fc_U(cond).view(B, self.rank, -1)
            vv = self.fc_V(cond).view(B, self.rank, -1)
        else:
            raise ValueError(v)
        self._u = u; self._v = vv

    def forward(self, t, h):
        dot = torch.einsum('bp,brp->br', h, self._v)
        return torch.einsum('br,brp->bp', dot, self._u)


# ---------------- 训练 / 预测 / 指标 (与 knode_dynamic_baselines_full_arab 同协议) ----------------
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


def train_variant(variant, Ys_tr, S_tr, y1_tr, seed, n_epochs, ode_rank, grm_dim):
    torch.manual_seed(seed); np.random.seed(seed)
    B, Tn, Pn = Ys_tr.shape
    m = CondVariantODE(Pn, grm_dim, Pn, ode_rank, variant).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    y_tr = torch.tensor(Ys_tr.astype(np.float32), device=DEVICE)
    g_tr = torch.tensor(S_tr, device=DEVICE)
    y1_t = torch.tensor(y1_tr.astype(np.float32), device=DEVICE)
    best = None; best_loss = float('inf')
    for ep in range(n_epochs):
        m.train()
        m.set_z(g_tr, y1_t)
        pred = sg.euler_integrate(m, y1_t, sg.times_t)
        loss = F.mse_loss(pred, y_tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 50 == 0:
            with torch.no_grad():
                m.eval()
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


def predict_variant(m, grm, y1n):
    with torch.no_grad():
        g = torch.tensor(np.asarray(grm, dtype=np.float32), device=DEVICE)
        y = torch.tensor(np.asarray(y1n, dtype=np.float32), device=DEVICE)
        m.set_z(g, y)
        pred = sg.euler_integrate(m, y, sg.times_t)
    return pred.cpu().numpy()


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
        m = train_variant(variant, Ys_tr, S_tr, y1_all[tr], seed, n_epochs, ode_rank, grm_dim)
        p_te_s.append(predict_variant(m, S_te, y1_all[te]))
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    p_te_s = [((p * tr_rng + tr_min - te_min) / te_rng).astype(np.float32) for p in p_te_s]
    p_te = np.mean(p_te_s, axis=0)
    r_all, _ = _r_nrmse(p_te, Y_te_n, np.arange(sg.T))
    r_tt, nm_tt = _r_nrmse(p_te, Y_te_n, obs)
    return {'r_含t0': r_all, 'r_剔t0': r_tt, 'nrmse_剔t0': nm_tt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='hyper',
                    choices=['baseline', 'hyper', 'attn', 'kernel', 'scale', 'genemb'])
    ap.add_argument('--seeds', default='42,1234,2024,2025')
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--iters', default='0')
    ap.add_argument('--rank', type=int, default=2)
    ap.add_argument('--grm_dim', type=int, default=40)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(',') if x.strip()]
    iters = [int(x) for x in a.iters.split(',') if x.strip()]
    sg.GRM_DIM = a.grm_dim
    print(f'Device:{DEVICE} variant={a.variant} seeds={seeds} epochs={a.epochs} '
          f'iters={iters} rank={a.rank} grm_dim={a.grm_dim}', flush=True)

    per_iter = {}
    for k in iters:
        print(f'===== iteration {k} variant={a.variant} =====', flush=True)
        per_iter[k] = run_iteration(a.variant, seeds, a.epochs, k, a.rank, a.grm_dim)
        print(f'  {a.variant:8s} | 含t0 r={per_iter[k]["r_含t0"]:.4f} '
              f'剔t0 r={per_iter[k]["r_剔t0"]:.4f} NRMSE剔t0={per_iter[k]["nrmse_剔t0"]:.4f}', flush=True)

    agg = {}
    for metric in ['r_含t0', 'r_剔t0', 'nrmse_剔t0']:
        vals = [per_iter[k][metric] for k in iters]
        agg[metric] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    tag = a.tag if a.tag else a.variant
    out_name = f'knode_cond_variants_arab_{tag}.json'
    with open(OUT_DIR / out_name, 'w', encoding='utf-8') as f:
        json.dump({'variant': a.variant, 'seeds': seeds, 'epochs': a.epochs, 'iters': iters,
                   'rank': a.rank, 'grm_dim': a.grm_dim, 'per_iter': per_iter, 'aggregate': agg},
                  f, ensure_ascii=False, indent=2, default=float)
    print(f'Saved: {out_name}', flush=True)


if __name__ == '__main__':
    main()
