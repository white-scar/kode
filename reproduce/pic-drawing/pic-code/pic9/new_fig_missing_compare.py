import os
# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
MP = "D:/pythonproject/knode/outputs/reproduce/missing_per_timepoint/"
KN = "D:/pythonproject/knode/outputs/reproduce/knode_missing_maize_arab/"

cols = {"krec":"#FF6B4A","kiter":"#C000C0","arec":"#1F77B4","aiter":"#2CA02C"}

def knode_series(ds, scen, mode, extra):
    df = pd.read_csv(KN + f"knode_missing_{ds}_v2_f10.csv")   # 10-fold (main method rank2+skip+affine)
    sub = df[(df.scenario==scen)&(df['mode']==mode)&(df.time.isin(extra))]
    return sub.groupby("time")["pcc"].mean()

def author_series(ds, scen, mode, extra):
    tag = {"maize":{"drop1":"drop1","drop2":"drop2","drop3":"drop3","delblock":"delblock"},
           "arab":{"drop_last1":"drop_last1","drop_last2":"drop_last2","drop_last3":"drop_last3","gap_mid":"gap_mid"}}[ds][scen]
    if ds=="maize":
        pref = "maize_" + tag
    else:
        pref = "arab_" + tag
    if mode=="rec":
        d = pd.read_csv(MP + f"{pref}_recursive_pcc_mean_sd.csv")
        d = d[d.Time.isin(extra)]
    else:
        d = pd.read_csv(MP + f"{pref}_iterative_extrapolation_pcc_mean_sd.csv")
        d = d[d.Time.isin(extra)]
    return d.set_index("Time")["Mean"]

def draw_panel(ax, ds, scen, extra, title):
    extra = sorted(extra)
    x = np.arange(len(extra), dtype=float)
    krec = knode_series(ds, scen, "rec", extra)
    kiter = knode_series(ds, scen, "iter", extra)
    arec = author_series(ds, scen, "rec", extra)
    aiter = author_series(ds, scen, "iter", extra)
    for mode, ser, col, mk, ls in [("krec",krec,cols["krec"],"o","-"),("kiter",kiter,cols["kiter"],"s","-"),
                                    ("arec",arec,cols["arec"],"o","--"),("aiter",aiter,cols["aiter"],"s","--")]:
        y = [ser.get(t, np.nan) for t in extra]
        ax.plot(x, y, color=col, ls=ls, lw=1.2, zorder=3)
        ax.plot(x, y, mk, color=col, ms=4, mec="white", mew=0.4, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([str(t) for t in extra], fontsize=6.5)
    ax.set_ylim(-0.1, 1.05); ax.set_yticks([0,0.5,1.0]); ax.tick_params(labelsize=6.5)
    ax.set_title(title, fontsize=8)
    ax.spines[["top","right"]].set_visible(False)

# ---- maize ----
fig, axes = plt.subplots(2, 2, figsize=(170/25.4, 150/25.4))
draw_panel(axes[0,0], "maize", "drop1", [47], "drop1 (missing DAS 47)")
draw_panel(axes[0,1], "maize", "drop2", [46,47], "drop2 (missing DAS 46-47)")
draw_panel(axes[1,0], "maize", "drop3", [45,46,47], "drop3 (missing DAS 45-47)")
draw_panel(axes[1,1], "maize", "delblock", [43,44,45,46,47], "delblock (missing DAS 43-47)")
axes[0,0].set_ylabel("Missing-point PCC", fontsize=8)
axes[1,0].set_ylabel("Missing-point PCC", fontsize=8)
handles = [Line2D([],[],color=cols[k],lw=1.3,marker="o" if k.endswith("rec") else "s",ms=4,mec="white",mew=0.4) for k in ["krec","arec","kiter","aiter"]]
labels = ["KNode rec","dynamicGP rec","KNode iter","dynamicGP iter"]
fig.legend(handles=handles, labels=labels, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5,0.01))
fig.tight_layout(rect=[0,0.05,1,1])
fig.savefig(OUT+"fig_missing_compare_maize.svg"); fig.savefig(OUT+"fig_missing_compare_maize.png", dpi=160)
plt.close(fig)

# ---- arab ----
fig, axes = plt.subplots(2, 2, figsize=(170/25.4, 150/25.4))
draw_panel(axes[0,0], "arab", "drop_last1", [21], "drop_last1 (missing DAS 21)")
draw_panel(axes[0,1], "arab", "drop_last2", [20,21], "drop_last2 (missing DAS 20-21)")
draw_panel(axes[1,0], "arab", "drop_last3", [19,20,21], "drop_last3 (missing DAS 19-21)")
draw_panel(axes[1,1], "arab", "gap_mid", [12,13,14,15,16], "gap_mid (missing DAS 12-16)")
axes[0,0].set_ylabel("Missing-point PCC", fontsize=8)
axes[1,0].set_ylabel("Missing-point PCC", fontsize=8)
fig.legend(handles=handles, labels=labels, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5,0.01))
fig.tight_layout(rect=[0,0.05,1,1])
fig.savefig(OUT+"fig_missing_compare_arab.svg"); fig.savefig(OUT+"fig_missing_compare_arab.png", dpi=160)
plt.close(fig)
print("saved fig_missing_compare_maize + fig_missing_compare_arab")
