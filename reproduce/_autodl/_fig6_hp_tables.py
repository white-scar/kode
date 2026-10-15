# -*- coding: utf-8 -*-
"""修正: 只排除 base_linear (drift=linear 的那份), 不要误排 act_linear"""
import glob, os
import pandas as pd
KD = "/root/autodl-tmp/qkn/outputs/kode_result"
OUT = "/root/autodl-tmp/qkn/_fig6_hp_tables"
os.makedirs(OUT, exist_ok=True)
SPEC = {"arab": ("arab", "knode_arab_tp1"),
        "rice": ("rice_indica_control", "knode_rice_tp1"),
        "uav":  ("maize_rgb_drought", "knode_maize_rgb_tp1")}
EXCL = {"base_linear"}
for ds, (sub, prefix) in SPEC.items():
    rows = []
    for f in sorted(glob.glob(os.path.join(KD, sub, prefix + "_all_summary_hp_*.csv"))):
        tag = os.path.basename(f).replace(prefix + "_all_summary_hp_", "").replace(".csv", "")
        if tag in EXCL or "smoke" in tag or tag.endswith("_time"):
            continue
        d = pd.read_csv(f)
        d = d[(d["split"] == "test") & (d["mode"] == "ode_hyper")]
        assert len(d) == 1, (f, len(d))
        r = d.iloc[0]
        rows.append(dict(config=tag, pcc=float(r["pcc"]), mse_norm=float(r["MSE_norm"])))
    t = pd.DataFrame(rows).sort_values("config")
    t.to_csv(os.path.join(OUT, "hyperparam_ablation_%s.csv" % ds), index=False)
    print("%-5s %2d config: %s" % (ds, len(t), list(t.config)))