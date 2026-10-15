# -*- coding: utf-8 -*-
"""make_fig_cond_ladder_data.py
把画图需要的所有数据从原始结果文件汇总成一个 CSV, 之后画图只读这一个文件。

输入 (原始结果, 每个 (数据集, 环境, D_G, rank) 一个独立 run directory):
    outputs/reproduce/cond_ladder_rice_dg10_r2/
        agg.csv            条件级 PCC/MSE/NRMSE/R2
        f0..f9.csv         逐折条件级 (用来算 MSE SD)
        trait_f0..f9.csv   逐 trait x timepoint x fold

输出 (本目录):
    fig_cond_ladder_rice_data.csv
    列: block, cond, trait, pcc, pcc_sd, mse, mse_sd, nrmse, r2
      block='overall' -> 每条件一行 (含 SD)
      block='trait'   -> 每 条件 x 性状 一行 (只有 pcc / mse)
"""
import os, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "outputs", "reproduce"))
OUT  = os.path.join(HERE, "fig_cond_ladder_rice_data.csv")
KEYS = ["persistence", "shared", "grm", "tp1", "full"]

# 结果目录: 2026-09-19 起每个 (数据集, 环境, D_G, rank) 一个独立 run directory;
#   'full' 档由阶梯脚本自己跑出来, 不再从主结果搬数。
RUN_DIR = os.path.join(RAW, "cond_ladder_rice_dg10_r2")     # 锁定配置: D_G=10, rank=2
if not os.path.isdir(RUN_DIR):
    cands = sorted(glob.glob(os.path.join(RAW, "cond_ladder_rice_dg*_r*")))
    assert len(cands) == 1, f"找不到 run directory: 候选={cands}; 请把 RUN_DIR 写成要画的那个"
    RUN_DIR = cands[0]
print("run dir ->", RUN_DIR)
assert os.path.isdir(RUN_DIR), f"run directory 不存在: {RUN_DIR}"

# ---------- 1) overall ----------
agg = pd.read_csv(os.path.join(RUN_DIR, "agg.csv")).set_index("cond")

fold_files = sorted(glob.glob(os.path.join(RUN_DIR, "f[0-9].csv")))
assert fold_files, "找不到 f[0-9].csv (run directory 里还没有结果?)"
fd = pd.concat([pd.read_csv(f) for f in fold_files], ignore_index=True)

overall = []
for k in KEYS:
    overall.append(dict(
        block="overall", cond=k, trait="",
        pcc=float(agg.loc[k, "pcc_mean"]),
        pcc_sd=float(agg.loc[k, "pcc_std"]),
        mse=float(agg.loc[k, "mse_mean"]),
        mse_sd=float(fd[fd["cond"] == k]["mse"].std(ddof=1)),
        nrmse=float(agg.loc[k, "nrmse_mean"]),
        r2=float(agg.loc[k, "r2_mean"]),
    ))
overall = pd.DataFrame(overall)

# ---------- 2) trait level ----------
trait_files = sorted(glob.glob(os.path.join(RUN_DIR, "trait_f[0-9].csv")))
assert trait_files, "找不到 trait_f[0-9].csv (run directory 里还没有结果?)"
det = pd.concat([pd.read_csv(f) for f in trait_files], ignore_index=True)
det = det.dropna(subset=["pcc", "mse"])

# 每个 (cond, trait): 先对 timepoint 平均, 再对 fold 平均
per = (det.groupby(["cond", "fold", "trait"])[["pcc", "mse"]].mean()
          .groupby(["cond", "trait"]).mean().reset_index())
per = per[per["cond"].isin(KEYS)]
trait = per.assign(block="trait", pcc_sd=np.nan, mse_sd=np.nan,
                   nrmse=np.nan, r2=np.nan)


out = pd.concat([overall, trait[["block", "cond", "trait", "pcc", "pcc_sd",
                                "mse", "mse_sd", "nrmse", "r2"]]],
                ignore_index=True)
out = out[["block", "cond", "trait", "pcc", "pcc_sd", "mse", "mse_sd", "nrmse", "r2"]]
out.to_csv(OUT, index=False)

print(f"saved -> {OUT}")
print(f"  rows = {len(out)}  (overall {len(overall)} + trait {len(trait)})")
print(f"  traits = {trait['trait'].nunique()}  conds = {sorted(trait['cond'].unique())}")
