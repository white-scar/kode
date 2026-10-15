#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""????? RGB ???? ?? ?????.

??(???????):
  1) ???: 37 ? VI ?????? (Red/Green/Blue); ?????? 35 ???
  2) ???: ???????????? (drought 522 / irrigated 520, ? bio_ID == Entry)
  3) Mantel: ?????"???x12???"?? -> ???????? -> ???? Pearson(Mantel r)
  4) ?? thr -> ?????? -> ??? h2 ?????
  5) ??????
??: ????????? (Mantel ?? / ???????)
"""
import numpy as np, pandas as pd, os, sys
import scipy.sparse as sp, scipy.sparse.csgraph as cs
from itertools import combinations

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MM   = os.path.join(BASE, 'data', 'maize_multispectral')
RGBD = os.path.join(BASE, 'data', 'maize_rgb')
FLIGHTS = [42, 56, 61, 64, 68, 71, 75, 78, 82, 84, 90, 99]
H2 = pd.read_csv(os.path.join(MM, 'maize_rgb_all_traits_h2_12tp.csv')).set_index('trait')

SPEC = {   # env -> (raw ??, ???, ?? Mantel, ????, h2 ?, ??????)
 'drought':   ('YYCD18_RGB.csv', 'Flight', '_tmp_mantel_maize_drought_rgb.npy',
               'maize_rgb_selected_traits_drought_rgb.csv', 'h2_drought',   'maize_rgb_drought_traits.csv'),
 'irrigated': ('YYCI18_RGB.csv', 'DAP',    '_tmp_mantel_maize_irrigated_rgb.npy',
               'maize_rgb_selected_traits_irrigated_rgb.csv', 'h2_irrigated', 'maize_rgb_irrigated_traits.csv'),
}


def load_env(env, drop_bands=True):
    fname, tcol, tmp, selcsv, hcol, traf = SPEC[env]
    hdr = pd.read_csv(os.path.join(MM, fname), nrows=0)
    VI = [c for c in hdr.columns[4:41] if c in set(H2.index)]      # ???? h2 ? VI
    if drop_bands:
        VI = [c for c in VI if c not in ('Red', 'Green', 'Blue')]
    d = pd.read_csv(os.path.join(MM, fname), usecols=['Entry', tcol] + VI)
    d[VI] = d[VI].apply(pd.to_numeric, errors='coerce')
    d['Entry'] = d['Entry'].astype(str)
    keep_ids = set(pd.read_csv(os.path.join(RGBD, traf)).bio_ID.astype(str))   # ?????????
    d = d[d['Entry'].isin(keep_ids) & d[tcol].isin(FLIGHTS)]
    g = d.groupby(['Entry', tcol])[VI].mean().reset_index()
    M = {}
    for t in VI:
        p = g.pivot(index='Entry', columns=tcol, values=t).reindex(columns=FLIGHTS)
        p = p.reindex(sorted(keep_ids & set(p.index)))            # ???????
        M[t] = p
    common = sorted(set.intersection(*[set(p.dropna().index) for p in M.values()]))
    V = {t: M[t].loc[common].values.astype(float) for t in VI}
    return V, VI, common, hcol


def mantel_matrix(V, VI):
    iu = np.triu_indices(V[VI[0]].shape[0], 1)
    Dm = {t: np.sqrt(((V[t][:, None, :] - V[t][None, :, :]) ** 2).sum(-1)) for t in VI}
    R = np.eye(len(VI))
    for i in range(len(VI)):
        for j in range(i + 1, len(VI)):
            R[i, j] = R[j, i] = float(np.corrcoef(Dm[VI[i]][iu], Dm[VI[j]][iu])[0, 1])
    return R


def select_from(R, VI, hcol, thr):
    A = (R > thr).astype(int)
    n, lab = cs.connected_components(sp.csr_matrix(A))
    sel, sizes = [], []
    for g in range(n):
        mem = [VI[k] for k in range(len(VI)) if lab[k] == g]
        sizes.append(len(mem))
        sel.append(max(mem, key=lambda t: H2.loc[t, hcol] if t in H2.index else -1))
    return sorted(sel), sorted(sizes, reverse=True)


if __name__ == '__main__':
    out = {}
    for env in ['drought', 'irrigated']:
        V, VI, gids, hcol = load_env(env)
        fname, tcol, tmp, selcsv, _, _ = SPEC[env]
        saved = set(pd.read_csv(os.path.join(MM, selcsv)).trait)
        R = mantel_matrix(V, VI)
        tmp_p = os.path.join(MM, tmp)
        sv = np.load(tmp_p) if os.path.exists(tmp_p) else R   # 缓存缺失时用现算矩阵
        print('=' * 78)
        print('[%s] ???=%d  ???=%d  ???=%d' % (env, len(VI), len(gids), sv.shape[0]))
        if sv.shape == R.shape:
            off = ~np.eye(len(VI), dtype=bool)
            print('   Mantel vs ??: max|?|=%.4f  mean|?|=%.4f  corr=%.4f' % (
                np.abs(R - sv)[off].max(), np.abs(R - sv)[off].mean(),
                np.corrcoef(R[off], sv[off])[0, 1]))
        print('   ???? %d ?: %s' % (len(saved), ', '.join(sorted(saved))))
        print('   thr   | ?? | ??? | ??????')
        best_thr, best = None, -1
        for thr in np.arange(0.94, 0.9951, 0.005):
            sel, sizes = select_from(R, VI, hcol, float(thr))
            hit = len(set(sel) & saved)
            flag = 'EXACT ?' if set(sel) == saved else ''
            print('   %.3f | %4d | %6d | %2d/%d %s' % (thr, len(sizes), len(sel), hit, len(saved), flag))
            if hit > best:
                best, best_thr, best_sel = hit, float(thr), sel
        print('   ??: thr=%.3f  ?? %d/%d' % (best_thr, best, len(saved)))
        print('     ??: %s' % ', '.join(best_sel))
        print('     ????: %s' % ', '.join(sorted(saved - set(best_sel))))
        print('     ????: %s' % ', '.join(sorted(set(best_sel) - saved)))
        out[env] = (best_thr, set(best_sel), saved)
    # ????
    print('=' * 78)
    i17 = set(pd.read_csv(os.path.join(MM, 'maize_rgb_selected_traits_intersection17.csv')).trait)
    for thr in sorted(set(v[0] for v in out.values())):
        inter = out['drought'][1] & out['irrigated'][1] if thr == out['drought'][0] else None
    our17 = out['drought'][1] & out['irrigated'][1]
    print('??????? = %d ? | ??? 17 ??? %d' % (len(our17), len(our17 & i17)))
    print('   ??:', ', '.join(sorted(our17)))
    print('   ??:', ', '.join(sorted(i17)))
    print('   ????:', sorted(i17 - our17), ' ????:', sorted(our17 - i17))
