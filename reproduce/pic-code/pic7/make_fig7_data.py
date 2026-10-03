# -*- coding: utf-8 -*-
"""make_fig7_data.py -- 图七数据: 四个数据集各自一份 (Kode 用今天 @1000 的新外推结果)
输出 (本目录):
  fig7_<ds>_b.csv   method, DAS, pcc_mean, pcc_sd, mse_mean, mse_sd, n_foldsets   (rec_tk, 尾部逐 DAS)
  fig7_<ds>_c.csv   method, foldset, time, trait, mse                            (rec_tk, 逐 trait)
  fig7_<ds>_d.csv   method, scenario, pcc_mean, pcc_sd, mse_mean, mse_sd, n_foldsets
作者方法: 只有玉米那张图里留有冻结值 (旧 fig7b/c/d), 这里原样并入 maize 的三张表; 其余数据集只有 Kode。
"""
import os
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                      # pic-code/pic7 -> knode
RD = ROOT / "outputs" / "reproduce"
SPECS = [("maize", "knode_extrap_maize_nt5", "v2_f10_nt5", "maize"),
         ("arab",  "knode_extrap_arab_nt2",  "v2_f10_nt2", "arab"),
         ("rice",  "knode_extrap_rice_nt2",  "v2_f10_nt2", "rice"),
         ("uav",   "knode_extrap_rgb_nt2",   "drought_v2_f10_nt2", "rgb")]
TAG_FULL = {"maize": "", "arab": "", "rice": "", "uav": "_drought"}   # fulltraj 文件名里的环境后缀
SCEN = "rec_tk"

for ds, sub, tag, filekey in SPECS:
    d = RD / sub
    ov = pd.read_csv(d / f"knode_extrap_{filekey}_{tag}_overall.csv")
    ft = pd.read_csv(d / f"knode_extrap_{filekey}_fulltraj_{tag}.csv")
    bt = pd.read_csv(d / f"knode_extrap_{filekey}_fulltraj_bytrait_{tag}.csv")
    # 2026-09-27: ?? b ?????"??? SD"(?????): ????? fold ??, ????? SD
    _pt = bt[bt.scenario == SCEN].groupby(["time", "trait"], as_index=False)[["pcc", "mse"]].mean()
    b = _pt.groupby("time").agg(
        pcc_mean=("pcc", "mean"), pcc_sd=("pcc", "std"),
        mse_mean=("mse", "mean"), mse_sd=("mse", "std")).reset_index()
    b["n_foldsets"] = ft[ft.scenario == SCEN].groupby("time")["foldset"].nunique().values
    b.insert(0, "method", "Kode"); b = b.rename(columns={"time": "DAS"})
    d0 = ov.groupby("scenario").agg(
        pcc_mean=("pcc", "mean"), pcc_sd=("pcc", "std"),
        mse_mean=("mse", "mean"), mse_sd=("mse", "std"), n_foldsets=("foldset", "nunique")).reset_index()
    d0.insert(0, "method", "Kode")
    tail = sorted(b["DAS"].unique())
    c = bt[(bt.scenario == SCEN) & (bt.time.isin(tail))][["foldset", "time", "trait", "mse"]].copy()
    c.insert(0, "method", "Kode")
    if ds == "maize":     # 作者方法: 并入旧图里冻结的 reference 值
        B0 = pd.read_csv(HERE / "fig7b_cv3_traj_pcc_mse.csv")
        C0 = pd.read_csv(HERE / "fig7c_cv3_bytrait_mse.csv")
        D0 = pd.read_csv(HERE / "fig7d_overall_scenarios.csv")
        au = sorted(set(B0.method) - {"Kode"})
        # 2026-09-27: ????????????? SD, ????? 3scen ?????
        #   (???????????????: iter 0.5087 / rec 0.3325 / rec_tk 0.4934)
        _A3 = pd.read_csv(ROOT / "outputs" / "reproduce" / "dynamicGP_MegaLMM_maize_CV3_3scen" / "snapshot_metrics_long.csv")
        _A3 = _A3[_A3.Method == "tp1_tk"]
        _tv = sorted(set(_A3.Time), key=lambda s: int(s[1:]))
        _ft = _tv[-len(B0):]
        _sub = _A3[_A3.Time.isin(_ft)]
        _pt = _sub.groupby(["Time", "Trait"], as_index=False)["PCC"].mean()
        _sd = _pt.groupby("Time")["PCC"].std().values
        _B0a = B0[B0.method.isin(au)].copy()
        _B0a = _B0a.sort_values("DAS")
        _B0a["pcc_sd"] = _sd
        b = pd.concat([b, _B0a], ignore_index=True)
        c = pd.concat([c, C0[C0.method.isin(au)]], ignore_index=True)
        d0 = pd.concat([d0, D0[D0.method.isin(au)]], ignore_index=True)
        print("  [maize] 并入作者冻结值:", au)
    # 2026-09-27: ????????? (maize ???, ????????)
    outdir = HERE if ds == "maize" else (HERE / ds)
    outdir.mkdir(exist_ok=True)
    b.to_csv(outdir / f"fig7_{ds}_b.csv", index=False)
    c.to_csv(outdir / f"fig7_{ds}_c.csv", index=False)
    d0.to_csv(outdir / f"fig7_{ds}_d.csv", index=False)
    print("%-5s -> fig7_%s_b/c/d.csv | b=%d 行 (DAS %s..%s) | c=%d 行 | d=%d 行"
          % (ds, ds, len(b), tail[0], tail[-1], len(c), len(d0)))
    print("      Kode 尾部 rec_tk PCC:", b[b.method == "Kode"][["DAS", "pcc_mean"]].round(4).to_dict("records"))
