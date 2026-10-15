#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_traits.py -- UAV-RGB: 原始 CSV -> Kode 性状长表（含性状选择）

流程对齐 Hobby et al. 的 cluster-and-select：
  0) 时间点质控（见 timepoint_qc()；分析代码在 data/maize_RGB/h2_by_flight.R）
     - 剔除缺失率 > 50% 的 flight          -> DAP 27, 34
     - 逐 flight x 性状 GREML 估 SNP-based h2, 对 34 个 RGB VI 取平均,
       平均 h2 < THR_H2 (=0.12) 的 flight 剔除 -> DAP 110, 127
     - 以 drought 为准, 结果时间点集原样用于 irrigated
     - 结果 = FLIGHTS (12 点): 42,56,61,64,68,71,75,78,82,84,90,99
  1) 取原始表第 4..41 列的 37 个 VI，去掉 Red/Green/Blue -> 34 个
  2) 按 Entry x Flight(DAP) 对重复植株取均值
  3) Mantel r = 两个性状距离矩阵的上三角 Pearson（等价 ade4::mantel.rtest$obs）
  4) r >= THR 连边 -> Louvain 聚类
  5) 每簇取 h2 最高的性状作代表
  6) 输出 (基因型 x 时间点) 长表

阈值 0.97、种子 42 固定。两个环境各选各的性状，不取交集。

输入（本目录）：
  YYCD18_RGB.csv / YYCI18_RGB.csv
  geno_GRM.csv
  ../maize_multispectral/maize_rgb_all_traits_h2_12tp.csv
输出（本目录）：
  rgb_selected_traits_{drought,irrigated}_thr097.csv
  maize_rgb_{drought,irrigated}_traits.csv
"""
import os
import numpy as np
import pandas as pd
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
THR = 0.97
SEED = 42
THR_H2 = 0.12            # 时间点质控: 跨性状平均 SNP-based h2 阈值
MAX_MISSING = 0.50       # 时间点质控: 表型缺失率上限
H2_CSV = os.path.join(HERE, "maize_rgb_h2_by_flight.csv")   # 由 h2_by_flight.R 生成
# FLIGHTS = 时间点质控(timepoint_qc) 的结果; 以 drought 为准, irrigated 原样套用
FLIGHTS = [42, 56, 61, 64, 68, 71, 75, 78, 82, 84, 90, 99]


def timepoint_qc(h2_csv=H2_CSV):
    """时间点质控 -> 保留的 flight 列表 + 明细表。

    规则 (与 data/maize_RGB/h2_by_flight.R 一致):
      (i)  缺失率 > MAX_MISSING 的 flight 剔除           -> DAP 27, 34
      (ii) 跨 34 个 RGB VI 平均的 SNP-based h2 < THR_H2  -> DAP 110, 127
    说明: h2 用 GREML (rrBLUP::mixed.solve(y, K)) 逐 flight x 性状估计;
          以 drought treatment 为准做筛选, 结果时间点集原样应用于 irrigated。
    """
    if not os.path.isfile(h2_csv):
        raise FileNotFoundError(
            "缺少 %s；请先在仓库根目录运行: Rscript data/maize_RGB/h2_by_flight.R" % h2_csv)
    tab = pd.read_csv(h2_csv)
    keep = [int(v) for v in tab.loc[tab.subset == "kept", "DAP"].tolist()]
    return keep, tab
SPEC = {
    "drought":   dict(raw="YYCD18_RGB.csv", tcol="Flight"),
    "irrigated": dict(raw="YYCI18_RGB.csv", tcol="DAP"),
}
H2 = pd.read_csv(os.path.join(HERE, "..", "maize_multispectral",
                              "maize_rgb_all_traits_h2_12tp.csv")).set_index("trait")
KEEP_IDS = set(pd.read_csv(os.path.join(HERE, "geno_GRM.csv"), index_col=0).index.astype(str))


def load_env(env):
    spec = SPEC[env]
    f = os.path.join(HERE, spec["raw"])
    tcol = spec["tcol"]
    hdr = pd.read_csv(f, nrows=0)
    VI = [c for c in hdr.columns[4:41]
          if c in set(H2.index) and c not in ("Red", "Green", "Blue")]
    d = pd.read_csv(f, usecols=["Entry", tcol] + VI)
    d["Entry"] = d["Entry"].astype(str)
    d = d[d["Entry"].isin(KEEP_IDS) & d[tcol].isin(FLIGHTS)]
    g = d.groupby(["Entry", tcol])[VI].mean().reset_index()
    V = {}
    for t in VI:
        V[t] = g.pivot(index="Entry", columns=tcol, values=t).reindex(columns=FLIGHTS)
    common = sorted(set.intersection(*[set(p.dropna().index) for p in V.values()]))
    return {t: V[t].loc[common].values.astype(float) for t in VI}, VI, common


def mantel_matrix(V, VI):
    iu = np.triu_indices(V[VI[0]].shape[0], 1)
    D = {t: np.sqrt(((V[t][:, None, :] - V[t][None, :, :]) ** 2).sum(-1)) for t in VI}
    R = np.eye(len(VI))
    for i in range(len(VI)):
        for j in range(i + 1, len(VI)):
            R[i, j] = R[j, i] = float(np.corrcoef(D[VI[i]][iu], D[VI[j]][iu])[0, 1])
    return R


def select_traits(R, VI, hcol):
    G = nx.Graph()
    G.add_nodes_from(range(len(VI)))
    for i in range(len(VI)):
        for j in range(i + 1, len(VI)):
            if R[i, j] >= THR:
                G.add_edge(i, j)
    comms = nx.algorithms.community.louvain_communities(G, seed=SEED)
    rows = []
    for c, members in enumerate(comms):
        names = [VI[k] for k in sorted(members)]
        vals = np.array([H2.loc[t, hcol] if t in H2.index else np.nan for t in names], dtype=float)
        pick = names[int(np.nanargmax(vals))]
        rows.append(dict(trait=pick, h2_mean=float(np.nanmax(vals)),
                         cluster_size=len(names), cluster=c))
    return pd.DataFrame(rows).sort_values("trait").reset_index(drop=True)


def main():
    summary = {}
    for env in ("drought", "irrigated"):
        V, VI, ids = load_env(env)
        R = mantel_matrix(V, VI)
        sel = select_traits(R, VI, "h2_%s" % env)
        sel.to_csv(os.path.join(HERE, "rgb_selected_traits_%s_thr097.csv" % env), index=False)

        parts = []
        for t in sel.trait:
            p = pd.DataFrame(V[t], index=ids, columns=FLIGHTS)
            s = p.stack().rename(t).reset_index()
            s.columns = ["bio_ID", "DAS", t]
            parts.append(s.set_index(["bio_ID", "DAS"]))
        wide = pd.concat(parts, axis=1).reset_index()
        wide.to_csv(os.path.join(HERE, "maize_rgb_%s_traits.csv" % env), index=False)

        # 同步输出该品系集的 GRM 子矩阵 (p 仍由全部 533 个基因型估计)
        grm_full = pd.read_csv(os.path.join(HERE, "geno_GRM.csv"), index_col=0)
        grm_full.index = grm_full.index.astype(str)
        grm_full.columns = grm_full.columns.astype(str)
        grm_full.loc[ids, ids].to_csv(os.path.join(HERE, "geno_GRM_%s.csv" % env))

        summary[env] = set(sel.trait)
        print("[%s] VI=%d  ids=%d  clusters=%d  selected=%d  -> maize_rgb_%s_traits.csv (%d x %d)"
              % (env, len(VI), len(ids), sel.cluster_size.count(), len(sel), env,
                 wide.shape[0], wide.shape[1]))
        print("   ", ", ".join(sel.trait))
    ov = summary["drought"] & summary["irrigated"]
    print("\noverlap %d | drought-only: %s | irrigated-only: %s"
          % (len(ov), ", ".join(sorted(summary["drought"] - summary["irrigated"])),
             ", ".join(sorted(summary["irrigated"] - summary["drought"]))))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", action="store_true",
                    help="只打印时间点质控结果(需 maize_rgb_h2_by_flight.csv), 不重建性状表")
    a = ap.parse_args()
    if a.qc:
        kept, tab = timepoint_qc()
        print(tab.to_string(index=False))
        print("\nkept flights:", kept)
        assert kept == FLIGHTS, "FLIGHTS 与 timepoint_qc() 的结果不一致!"
        print("与脚本内 FLIGHTS 校验通过")
    else:
        main()