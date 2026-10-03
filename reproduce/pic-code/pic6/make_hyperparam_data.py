# -*- coding: utf-8 -*-
"""make_hyperparam_data.py
把超参消融的原始汇总整理成 pic6 画图要用的 hyperparam_ablation_<ds>.csv。
输入 (既有结果, 不重跑):
  maize : outputs/reproduce/hyperparam_ablation_maize_current.csv     (已是目标格式, 直接复制)
  arab  : outputs/reproduce/hyperparam_ablation_arab_summary.csv      (取 mode=ode_hyper & split=test)
  rice  : outputs/reproduce/hyperparam_ablation_rice_summary.csv      (同上)
  uav   : outputs/reproduce/hyperparam_ablation_uav_summary.csv       (同上)
输出:
  pic6/hyperparam_ablation.csv            (maize)
  pic6/arab/hyperparam_ablation_arab.csv
  pic6/rice/hyperparam_ablation_rice.csv
  pic6/uav/hyperparam_ablation_uav.csv
  列: config, pcc, mse_norm
"""
import os, shutil
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
R    = os.path.join(ROOT, "outputs", "reproduce")

# maize: 直接复制目标格式文件
shutil.copyfile(os.path.join(R, "hyperparam_ablation_maize_current.csv"),
                os.path.join(HERE, "hyperparam_ablation.csv"))
print("saved pic6/hyperparam_ablation.csv (copy)")

SUB = {"arab": "arab", "rice": "rice", "uav": "uav"}
for ds, sub in SUB.items():
    d = pd.read_csv(os.path.join(R, "hyperparam_ablation_%s_summary.csv" % ds))
    d = d[(d["mode"] == "ode_hyper") & (d["split"] == "test")]
    out = d[["config", "pcc", "MSE_norm"]].rename(columns={"MSE_norm": "mse_norm"})
    p = os.path.join(HERE, sub, "hyperparam_ablation_%s.csv" % ds)
    out.to_csv(p, index=False, lineterminator="\n")   # 统一 LF
    print("saved", os.path.relpath(p, ROOT), "| rows =", len(out))
