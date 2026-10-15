# -*- coding: utf-8 -*-
"""_cv3_agg.py <arab|rice> <k0> <k1> -- 把每折的 snapshot_metrics_long.csv 汇总成整份结果"""
import sys
from pathlib import Path
import pandas as pd

DS, K0, K1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
TAG = "arabidopsis" if DS == "arab" else "rice"
BASE = Path("/root/autodl-tmp/qkn/outputs/reproduce/dynamicGP_MegaLMM_%s_CV3_unified" % TAG)
fs = [BASE / ("fold%d" % k) / "snapshot_metrics_long.csv" for k in range(K0, K1 + 1)]
fs = [f for f in fs if f.is_file()]
assert fs, "没有找到任何 fold*/snapshot_metrics_long.csv"
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
d.to_csv(BASE / "snapshot_metrics_long.csv", index=False)
by = d.groupby(["Method", "Time"], as_index=False).agg(PCC=("PCC", "mean"), MSE=("MSE", "mean"))
by.to_csv(BASE / "snapshot_metrics_by_time.csv", index=False)
ov = d.groupby("Method", as_index=False).agg(PCC=("PCC", "mean"), MSE=("MSE", "mean"), n=("PCC", "size"))
ov.to_csv(BASE / "snapshot_metrics_overall.csv", index=False)
print("聚合 %d 折 -> %s" % (len(fs), BASE))
print(by.pivot(index="Time", columns="Method", values="PCC").round(4).to_string())
print()
print(ov.round(4).to_string(index=False))