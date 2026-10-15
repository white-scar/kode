# -*- coding: utf-8 -*-
"""baselines_per_trait_foldavg.py - ? per_trait ? 10 ?(iter)?????

??: outputs/reproduce/knode_dynamic_baselines_full[_arab]_per_trait.csv
      (?: fam, iter, time, trait, pcc, mse, R2, NMSE, RMSE, NRMSE, y_true, y_pred)
??: outputs/reproduce/knode_dynamic_baselines_full[_arab]_per_trait_foldavg.csv
      (?: fam, time, trait, pcc, mse, R2, NMSE, RMSE, NRMSE, y_true, y_pred, SD)
  - ?? = 5 fam ? (T-1) ?? ? P ??  (maize: 5 ? 24 ? 50 = 6000)
  - 6 ???? + y_true/y_pred ?? 10 ?????
  - ?? SD = pcc ? 10 ???????? (ddof=1)
"""
import sys
import pandas as pd

BASE = "/data/fandata/dashuaige/outputs/reproduce/"
DS = sys.argv[1] if len(sys.argv) > 1 else "maize"
src = BASE + ("knode_dynamic_baselines_full_per_trait.csv" if DS == "maize"
              else "knode_dynamic_baselines_full_%s_per_trait.csv" % DS)
dst = src.replace("_per_trait.csv", "_per_trait_foldavg.csv")

d = pd.read_csv(src)
print("in :", src, d.shape)
means = ["pcc", "mse", "R2", "NMSE", "RMSE", "NRMSE", "y_true", "y_pred"]
g = d.groupby(["fam", "time", "trait"], as_index=False)[means].mean()
sd = (d.groupby(["fam", "time", "trait"])["pcc"].std(ddof=1)
        .rename("SD").reset_index())
out = g.merge(sd, on=["fam", "time", "trait"])
out = out[["fam", "time", "trait"] + means + ["SD"]]
out.to_csv(dst, index=False)
print("out:", dst, out.shape, "| cols =", list(out.columns))
print(out.head(3).to_string(index=False))
