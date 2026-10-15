# -*- coding: utf-8 -*-
"""unified_norm_eval_rice.py — 水稻(indica, control): 统一归一化口径下, 作者两个变体 与 Kode 的逐点 PCC/MSE

口径 (与 pic2/unified_norm_eval.py、arab-E1 完全一致):
  1) scaler 只在训练折(folds1-5)拟合: 每性状 min/max 取训练折内(基因型 x 时间)极值
  2) 训练/验证/验证真值共用同一套 scaler -> MSE 天然同空间
  3) 作者 DMD 用 scale='none' (禁用其内部二次 min-max)

SD 定义 (与 pic2_maize.R / pic3_maize.R 一致):
  _sd      = pooled SD: 对 (trait x fold) 池化后求 std   <-- 主用
  _sd_fold = 先折内对 trait 平均、再对折求 std           <-- 备用

作者变体 (服务器 author_rice_postproc.py 产出, 本地副本 author_rice_long.csv):
  tp1  = DynamicGP-MegaLMM+TP1  -> inner-5 集成
  mega = DynamicGP-MegaLMM      -> inner-5 集成

输出 (本目录):
  unified_norm_author_vs_kode_per_timepoint.csv
  unified_norm_author_vs_kode_per_foldset.csv
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"D:\pythonproject\knode"
DATA = os.path.join(BASE, "data", "rice_salinity")
OUT  = os.path.join(BASE, "outputs", "kode_result", "rice_indica_control")

# ---------------- 时间点 / 性状 ----------------
pheno = pd.read_csv(os.path.join(DATA, "rice_indica_control_traits.csv"))
pheno = pheno[pheno["DAS"] >= 30].copy()      # v2: 只用连续段 DAS 30-42 (丢掉 DAS28)
pheno["bio_ID"] = pheno["bio_ID"].astype(str)
times = sorted(pheno["DAS"].unique())
print(f"rice control: DAS={times}  n_traits={sum(1 for c in pheno.columns if c not in ("Unnamed: 0","bio_ID","accession","DAS"))}")

# ---------------- 作者 tp1 / mega: 逐 trait x fold 长表 ----------------
# 2026-09-27: mega now comes from the inner-5 ensemble script (outputs/dynamicgp_megalmm_rice_innerens);
#             tp1 is still taken from the old pipeline author_rice_long_v2.csv
_AUT_raw = pd.read_csv(os.path.join(HERE, "author_rice_long_v2.csv"))
_AUT_raw = _AUT_raw[_AUT_raw["variant"] == "tp1"].copy()
_dasmap = {f"t{i+1}": int(d) for i, d in enumerate(times)}
def _snap_to_aut(path, variant, methods):
    d = pd.read_csv(path)
    d = d[d["Method"].isin(methods)].copy()
    return pd.DataFrame(dict(
        variant=variant,
        scenario=np.where(d["Method"].str.endswith("_iter"), "iter", "rec"),
        foldset=d["Iteration"].values,
        timepoint=d["Time"].map(_dasmap).values,
        trait=d["Trait"].values,
        pcc=d["PCC"].values,
        MSE_norm=d["MSE"].values,
        R2=d["R2"].values,
        NRMSE=np.sqrt(d["NMSE"].values),
    ))

AUT = pd.concat([
    _AUT_raw,   # tp1: old pipeline (author_rice_long_v2.csv)
    _snap_to_aut(os.path.join(BASE, "outputs", "dynamicgp_megalmm_rice_innerens", "snapshot_metrics_long.csv"),
                 "mega", ["mega_rec", "mega_iter"]),
    _snap_to_aut(os.path.join(BASE, "outputs", "dynamicgp_rr_innerens_rice", "snapshot_metrics_long.csv"),
                 "rr", ["rr_rec", "rr_iter"]),
], ignore_index=True)
AUT = AUT[["variant","scenario","foldset","timepoint","trait","pcc","MSE_norm","R2","NRMSE"]].copy()
AUT = AUT.rename(columns={"foldset":"Fold","timepoint":"DAS","pcc":"PCC","MSE_norm":"MSE"})
AUT["NMSE"] = 1.0 - AUT["R2"]          # 与作者 R 脚本一致: NMSE = MSE/var = 1 - R2
AUT["Series"] = "author_" + AUT["variant"] + "_" + AUT["scenario"]
AUT = AUT[["Series","Fold","DAS","trait","PCC","MSE","R2","NMSE"]]
AUT = AUT[AUT.DAS != times[0]]          # 剔除 t1

# ---------------- 本方法: 逐 trait x fold ----------------
kd = []
for mode, prot in [("ode_hyper","rec"), ("iter_hyper","iter")]:
    d = pd.read_csv(os.path.join(OUT, f"knode_rice_tp1_{mode}_hp_base_per_trait.csv"))   # v2: 今天的主方法 (dim10/DAS>=30)
    d = d[(d.split=="test") & (d.timepoint!=times[0])]
    kd.append(pd.DataFrame(dict(Series=f"Kode_{prot}", Fold=d.foldset.values,
                                DAS=d.timepoint.values, trait=d.trait.values,
                                PCC=d.pcc.values, MSE=d.MSE_norm.values,
                                R2=d.R2.values, NMSE=(1.0 - d.R2.values))))
KD = pd.concat(kd, ignore_index=True)

# ---------------- 汇总 ----------------
fold = pd.concat([AUT, KD], ignore_index=True)
fold = fold[["Series","Fold","DAS","trait","PCC","MSE","R2","NMSE"]].sort_values(["Series","DAS"])
fold.to_csv(os.path.join(HERE, "unified_norm_author_vs_kode_per_foldset_v2.csv"), index=False)

SERIES_ORDER = ["author_tp1_rec","author_tp1_iter","author_mega_rec","author_mega_iter",
                "author_rr_rec","author_rr_iter","Kode_rec","Kode_iter"]
tab = pd.DataFrame({"DAS": [int(t) for t in times[1:]]})
for s in SERIES_ORDER:
    sub = fold[fold.Series==s]
    pooled = sub.groupby("DAS")[["PCC","MSE","R2","NMSE"]].agg(["mean","std"])
    foldm  = sub.groupby(["DAS","Fold"])[["PCC","MSE","R2","NMSE"]].mean()
    foldsd = foldm.groupby("DAS")[["PCC","MSE","R2","NMSE"]].std()
    for metric in ("PCC","MSE","R2","NMSE"):
        tab[f"{s}_{metric}"]         = [pooled.loc[t,(metric,"mean")] for t in tab.DAS]
        tab[f"{s}_{metric}_sd"]      = [pooled.loc[t,(metric,"std")]  for t in tab.DAS]
        tab[f"{s}_{metric}_sd_fold"] = [foldsd.loc[t,  metric]        for t in tab.DAS]

tab.to_csv(os.path.join(HERE, "unified_norm_author_vs_kode_per_timepoint_v2.csv"), index=False)

print("=== 均值 (pooled SD / fold SD) ===")
for s in SERIES_ORDER:
    print("%-18s PCC=%.4f (%.4f / %.4f)  MSE=%.5f (%.5f / %.5f)" % (
        s, tab[s+"_PCC"].mean(), tab[s+"_PCC_sd"].mean(), tab[s+"_PCC_sd_fold"].mean(),
        tab[s+"_MSE"].mean(), tab[s+"_MSE_sd"].mean(), tab[s+"_MSE_sd_fold"].mean()))
print("columns:", len(tab.columns), "| folded rows:", len(fold))
print("saved ->", os.path.join(HERE, "unified_norm_author_vs_kode_per_timepoint_v2.csv"))
