# -*- coding: utf-8 -*-
"""_ststp_single_compare.py - 50 个单性状 ST-STP vs E1 all 的 ST-STP 逐性状/逐时点公平对照

读: <root>/outputs/reproduce/knode_ststp_single_maize/single_p*/metrics_by_trait_time.csv
    <root>/outputs/reproduce/knode_tp1only_maize/all/metrics_by_trait_time.csv   (上一轮 E1 all)
写: <root>/outputs/reproduce/knode_ststp_single_maize/
      comparison_single_vs_all_ststp_by_trait.csv       (50 行)
      comparison_single_vs_all_ststp_by_trait_time.csv  (1200 行)
      comparison_summary.md
"""
import os
from pathlib import Path
import numpy as np, pandas as pd

import argparse
ROOT = Path(os.environ.get("KNODE_ROOT", "..")).resolve()
DS_CFG = {"maize": "knode_ststp_single_maize", "rice": "knode_ststp_single_rice", "arab": "knode_ststp_single_arab"}
_a = argparse.ArgumentParser(); _a.add_argument("--dataset", default="maize", choices=list(DS_CFG)); _A = _a.parse_args()
OUT = ROOT / "outputs" / "reproduce" / DS_CFG[_A.dataset]
E1 = ROOT / "outputs" / "reproduce" / ("knode_tp1only_maize" if _A.dataset == "maize"
                                      else ("knode_tp1only_rice" if _A.dataset == "rice" else "knode_tp1only_arab")) / "all" / "metrics_by_trait_time.csv"
TIE = 0.01  # |dPCC| < TIE 记为近似持平(描述性)
DS_NAME = {"maize": "\u7389\u7c73 MAGIC", "rice": "\u6c34\u7a3b indica", "arab": "\u62df\u5357\u82a5"}[_A.dataset]

M = ["pcc", "mse", "RMSE", "R2", "NMSE", "NRMSE", "y_true", "y_pred", "SD"]

def load_single():
    rows = []
    for d in sorted(OUT.glob("single_p*")):
        f = d / "metrics_by_trait_time.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        t = t[t.fam == "ststp"].copy()
        if t.empty:
            continue
        t["_dir"] = d.name
        rows.append(t[["trait", "time", "_dir"] + M])
    return pd.concat(rows, ignore_index=True)

def main():
    e1 = pd.read_csv(E1)
    e1 = e1[e1.fam == "ststp"][["trait", "time"] + M].copy()
    e1 = e1.rename(columns={c: c + "_all" for c in M})
    S = load_single()
    S = S.rename(columns={c: c + "_single" for c in M})
    m = S.merge(e1, on=["trait", "time"], how="inner")
    m["dPCC"] = m.pcc_single - m.pcc_all
    m["dMSE"] = m.mse_single - m.mse_all
    m["dRMSE"] = m.RMSE_single - m.RMSE_all
    m["dR2"] = m.R2_single - m.R2_all
    cols = (["trait", "time", "_dir"] +
            ["pcc_single", "pcc_all", "dPCC", "mse_single", "mse_all", "dMSE",
             "RMSE_single", "RMSE_all", "dRMSE", "R2_single", "R2_all", "dR2",
             "NMSE_single", "NMSE_all", "NRMSE_single", "NRMSE_all",
             "y_true_single", "y_pred_single", "y_true_all", "y_pred_all", "SD_single", "SD_all"])
    m = m[cols].sort_values(["trait", "time"])
    m.to_csv(OUT / "comparison_single_vs_all_ststp_by_trait_time.csv", index=False)

    g = m.groupby("trait", as_index=False).agg(
        n_time=("time", "nunique"),
        pcc_single=("pcc_single", "mean"), pcc_all=("pcc_all", "mean"),
        mse_single=("mse_single", "mean"), mse_all=("mse_all", "mean"),
        RMSE_single=("RMSE_single", "mean"), RMSE_all=("RMSE_all", "mean"),
        R2_single=("R2_single", "mean"), R2_all=("R2_all", "mean"),
        NRMSE_single=("NRMSE_single", "mean"), NRMSE_all=("NRMSE_all", "mean"))
    g["dPCC"] = g.pcc_single - g.pcc_all
    g["dMSE"] = g.mse_single - g.mse_all
    g["dRMSE"] = g.RMSE_single - g.RMSE_all
    g["dR2"] = g.R2_single - g.R2_all
    g["verdict"] = np.where(g.dPCC > TIE, "提升", np.where(g.dPCC < -TIE, "下降", "持平"))
    g = g.sort_values("dPCC", ascending=False)
    g.to_csv(OUT / "comparison_single_vs_all_ststp_by_trait.csv", index=False)

    # 后期（最后 1/4 时点）是否改善
    times = sorted(m.time.unique()); late = set(times[int(len(times) * 0.75):])
    gl = m[m.time.isin(late)].groupby("trait").dPCC.mean()
    n_up = int((g.verdict == "提升").sum()); n_tie = int((g.verdict == "持平").sum()); n_dn = int((g.verdict == "下降").sum())
    lines = []
    lines.append("# ST-STP \u5355\u6027\u72b6 vs \u5168\u6027\u72b6\uff08%s\uff09\u9010\u6027\u72b6\u5bf9\u7167\n" % DS_NAME)
    lines.append("> \u6570\u636e\uff1a%s \u540c\u4e00\u4efd\u8868\u578b\u3001\u540c\u4e00 CV\u3001iters=0,1,2\u3001inner-5\u3001seed=42\u3001\u53ea\u7528 TP1\uff08\u65e0 GRM\uff09\uff1b\u4ec5 ST-STP\u3002" % DS_NAME)
    lines.append("> \u6bd4\u8f83\uff1a**\u9010\u6027\u72b6\u4e00\u4e00\u5339\u914d**\uff08\u540c\u4e00 trial \u7684 %d \u4e2a\u672a\u6765\u771f\u5b9e\u89c2\u6d4b\u65f6\u70b9\uff09\uff0c\u4e0d\u62ff\u5355\u6027\u72b6 PCC \u53bb\u6bd4\u5168 %d \u6027\u72b6\u7684\u603b\u4f53\u5747\u503c\u3002" % (len(times), len(g)))
    lines.append("> 持平阈值：|ΔPCC| < %.2f（描述性，不作显著性检验；3 个 iteration 不足以声称统计显著）。\n" % TIE)
    lines.append("## 总体（%d 个性状）\n" % len(g))
    lines.append("| 判定 | 数量 | 占比 |")
    lines.append("|---|---|---|")
    lines.append("| 提升 (ΔPCC > +%.2f) | %d | %.0f%% |" % (TIE, n_up, 100.0 * n_up / len(g)))
    lines.append("| 持平 (|ΔPCC| < %.2f) | %d | %.0f%% |" % (TIE, n_tie, 100.0 * n_tie / len(g)))
    lines.append("| 下降 (ΔPCC < -%.2f) | %d | %.0f%% |\n" % (TIE, n_dn, 100.0 * n_dn / len(g)))
    lines.append("- ΔPCC：均值 %+.4f，中位 %+.4f，范围 %+.4f ~ %+.4f" %
                 (g.dPCC.mean(), g.dPCC.median(), g.dPCC.min(), g.dPCC.max()))
    lines.append("- ΔMSE（越低越好）：均值 %+.5f，中位 %+.5f；ΔMSE<0 的性状 %d 个" %
                 (g.dMSE.mean(), g.dMSE.median(), int((g.dMSE < 0).sum())))
    lines.append("- \u540e\u671f\uff08\u6700\u540e %d \u4e2a\u65f6\u70b9\uff09\u0394PCC \u5747\u503c %+.4f\uff1b\u5176\u4e2d\u6539\u5584\u7684\u6027\u72b6 %d \u4e2a\n" %
                 (len(late), gl.mean(), int((gl > 0).sum())))
    lines.append("## 提升最多（前 8）\n")
    lines.append("| trait | PCC_single | PCC_all | ΔPCC | ΔMSE | 判定 |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in g.head(8).iterrows():
        lines.append("| `%s` | %.4f | %.4f | %+.4f | %+.5f | %s |" % (r.trait, r.pcc_single, r.pcc_all, r.dPCC, r.dMSE, r.verdict))
    lines.append("\n## 下降最多（后 8）\n")
    lines.append("| trait | PCC_single | PCC_all | ΔPCC | ΔMSE | 判定 |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in g.tail(8).iloc[::-1].iterrows():
        lines.append("| `%s` | %.4f | %.4f | %+.4f | %+.5f | %s |" % (r.trait, r.pcc_single, r.pcc_all, r.dPCC, r.dMSE, r.verdict))
    lines.append("\n## 备注\n")
    lines.append("- \u5355\u6027\u72b6\u6a21\u578b\u4e0e E1 all \u4f7f\u7528**\u76f8\u540c\u7684 outer test IDs\u3001\u76f8\u540c %d \u4e2a\u672a\u6765\u65f6\u70b9\u3001\u76f8\u540c\u8bad\u7ec3\u6298 min-max**\uff08\u5404\u81ea\u6027\u72b6\u7684 min-max\uff09\u3002" % len(times))
    lines.append("- `ststp` 为静态多时点预测头（TP1 → 全部时点一次读出），不是逐步递归动力学模型。")
    lines.append("- \u660e\u7ec6\u89c1 `comparison_single_vs_all_ststp_by_trait_time.csv`\uff08%d \u884c\uff09\uff1b\u9010\u6027\u72b6\u6c47\u603b\u89c1 `..._by_trait.csv`\uff08%d \u884c\uff09\u3002" % (len(m), len(g)))
    (OUT / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved 3 comparison files to", OUT)
    print(g.verdict.value_counts().to_dict())
    print("dPCC mean %.4f median %.4f" % (g.dPCC.mean(), g.dPCC.median()))

if __name__ == "__main__":
    main()
