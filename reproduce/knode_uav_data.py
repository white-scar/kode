"""knode_uav_data.py — 无人机玉米(RGB, 干旱)数据模块, 镜像 sg 接口.
供 UAV 的 rank / 基线等脚本使用.
数据:
  - pheno: data/maize_rgb/maize_rgb_drought_traits.csv (17 性状, DAS 42-99)
  - folds: data/maize_rgb/maize_rgb_drought_cv_10fold.csv (522×10)
  - GRM  : data/maize_rgb/geno_GRM.csv (533×533, 带 ID)
与玉米/拟南芥 sg 模块唯一差别: DATA_DIR / 文件名 / PCA 维度, 其余逐行一致.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.decomposition import PCA
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.001; WEIGHT_DECAY = 1e-4
OUT_DIR = Path('../outputs/reproduce')
DATA_DIR = Path('../data/maize_rgb')

pheno = pd.read_csv(DATA_DIR / 'maize_rgb_drought_traits.csv')
cv_raw = pd.read_csv(DATA_DIR / 'maize_rgb_drought_cv_10fold.csv', index_col=0)
grm_full = pd.read_csv(DATA_DIR / 'geno_GRM.csv', index_col=0)

meta_cols = ['bio_ID', 'accession', 'DAS']
trait_cols = [c for c in pheno.columns if c not in meta_cols
              and not c.startswith('Unnamed') and c != '']
P = len(trait_cols)
times_all = np.array(sorted(pheno['DAS'].unique()), dtype=np.float64)
T = len(times_all)

common_ids = sorted(set(pheno['bio_ID'].astype(str)) & set(cv_raw.index))
N = len(common_ids)
acc = None  # 无人机表无 accession 列
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
GRM_DIM = 50  # 与无人机主方法一致
# baseline 脚本仍可用 --grm_dim 改写 (取前 GRM_DIM 个 PC; PCA 在模块级 fit 一次 50 维).


def get_S(dim=None):
    """返回前 dim 个 GRM 主成分 (默认 GRM_DIM). 取前 d 个 PC 等价于重新 fit PCA(d)."""
    d = int(GRM_DIM if dim is None else dim)
    return S_all[:, :d].astype(np.float32)


# ---------------- ODE 族 (rank-K Koopman, 默认 rank-1, 可选 GRM) ----------------
class ODEModel(nn.Module):
    """低秩 Koopman ODE: dh/dt = U (V^T h), U,V 为 P×K (rank=K), 由 cond 生成.
    rank=1 时退化为 u·(v·h) (原作者 KNode). rank>=2 提供 K 个指数模态."""
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
        # (V^T h)_r = <h, v_r>;  dh/dt = sum_r u_r * (V^T h)_r
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


if __name__ == '__main__':
    print(f'Device: {DEVICE}  N={N} T={T} P={P}  GRM PCA={pca.explained_variance_ratio_.sum():.3f}')
