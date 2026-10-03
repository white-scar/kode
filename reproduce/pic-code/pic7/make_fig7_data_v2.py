# -*- coding: utf-8 -*-
"""make_fig7_data_v2.py -- 图七 v2: 在 Kode 基础上并入【新跑的作者 CV3】数值 (arab / rice)
作者方法: tp1+MegaLMM+dynamicGP, 训练折 min-max 空间 (与 Kode 同一口径) -> PCC/MSE 可直接比
输出: fig7_<ds>_{b,c,d}_v2.csv (旧的不动)
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RD = ROOT / "outputs" / "reproduce"
AUTH = "dynamicGP-MegaLMM+TP1"
SPECS = [("arab", "arabidopsis", 8, 2), ("rice", "rice", 30, 2)]   # ds, CV3 目录后缀, 序列起始 DAS, n_test
SCEN_MAP = {"tp1_rec_t1": "rec_t1", "tp1_rec_tk": "rec_tk", "tp1_iter_t1": "iter"}

for ds, ctag, t0, nt in SPECS:
    _dd = HERE / ds   # 2026-09-27: ??????????
    b0 = pd.read_csv(_dd / f"fig7_{ds}_b.csv")
    _dd = HERE / ds   # 2026-09-27: ??????????
    c0 = pd.read_csv(_dd / f"fig7_{ds}_c.csv")
    _dd = HERE / ds   # 2026-09-27: ??????????
    d0 = pd.read_csv(_dd / f"fig7_{ds}_d.csv")
    L = pd.read_csv(RD / f"dynamicGP_MegaLMM_{ctag}_CV3_unified" / "snapshot_metrics_long.csv")
    kode_das0 = sorted(b0[b0.method == "Kode"]["DAS"].astype(int).unique())
    anchor = kode_das0[-(nt + 1)]                     # 锚点 DAS (尾部留出前的最后一个训练点)
    idx = L["Time"].str.replace("t", "", regex=False).astype(int)
    # *_t1 组: 时间标签从序列起点起算; *_tk 组: 从锚点 DAS 起算
    L["DAS"] = idx + (anchor - 1)
    is_t1 = L["Method"].str.endswith("_t1")
    L.loc[is_t1, "DAS"] = idx[is_t1] + (t0 - 1)
    kode_das = sorted(b0[b0.method == "Kode"]["DAS"].astype(int).unique())
    tail = kode_das[-nt:]
    print(f"[{ds}] 尾部 DAS={tail} | CV3 DAS 范围 {L.DAS.min()}..{L.DAS.max()}")

    A = L[(L.Method == "tp1_rec_tk") & (L.DAS.isin(tail))]
    assert len(A) > 0
    # 2026-09-27: ???????? SD (????? fold ??, ????? SD)
    _pt = A.groupby(["DAS", "Trait"], as_index=False)[["PCC", "MSE"]].mean()
    ab = _pt.groupby("DAS").agg(pcc_mean=("PCC", "mean"), pcc_sd=("PCC", "std"),
                                mse_mean=("MSE", "mean"), mse_sd=("MSE", "std")).reset_index()
    ab["n_foldsets"] = A.groupby("DAS")["Iteration"].nunique().values
    ab.insert(0, "method", AUTH)
    ac = A[["Iteration", "DAS", "Trait", "MSE"]].rename(columns={"Iteration": "foldset", "DAS": "time",
                                                                 "Trait": "trait", "MSE": "mse"})
    ac.insert(0, "method", AUTH)
    rows = []
    tails = L[L.DAS.isin(tail)]
    for fold, g in tails.groupby("Iteration"):
        for meth, scen in SCEN_MAP.items():
            s = g[g.Method == meth]
            rows.append(dict(foldset=fold, scenario=scen, pcc=s.PCC.mean(), mse=s.MSE.mean()))
    G = pd.DataFrame(rows).groupby("scenario").agg(
        pcc_mean=("pcc", "mean"), pcc_sd=("pcc", "std"),
        mse_mean=("mse", "mean"), mse_sd=("mse", "std"),
        n_foldsets=("foldset", "nunique")).reset_index()
    G.insert(0, "method", AUTH)

    b = pd.concat([b0, ab], ignore_index=True)
    c = pd.concat([c0, ac], ignore_index=True)
    d = pd.concat([d0, G], ignore_index=True)
    b.to_csv(_dd / f"fig7_{ds}_b_v2.csv", index=False)
    c.to_csv(_dd / f"fig7_{ds}_c_v2.csv", index=False)
    d.to_csv(_dd / f"fig7_{ds}_d_v2.csv", index=False)
    print("  作者 rec_tk 尾部:", ab[["DAS", "pcc_mean", "mse_mean"]].round(4).to_dict("records"))
    print("  作者 overall:", G.round(4).to_dict("records"))
