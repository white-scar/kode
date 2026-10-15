import os
# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = "D:/pythonproject/knode/outputs/reproduce/missing_per_timepoint/"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/"

def load(ds, scen):
    r = pd.read_csv(D + f"{ds}_{scen}_recursive_pcc_mean_sd.csv")
    ie = pd.read_csv(D + f"{ds}_{scen}_iterative_extrapolation_pcc_mean_sd.csv")
    return r, ie

fig, axes = plt.subplots(1, 2, figsize=(220/25.4, 90/25.4))

# ---- maize: delblock (DAS 43-47), rec vs iter, plus drop1/2/3 tail markers ----
ax = axes[0]
r, ie = load("maize", "delblock")
x = r["Time"].values[-5:]  # 43..47
rec = r["Mean"].values[-5:]
it = ie["Mean"].values
ax.plot(range(5), rec, color="#FF6B4A", ls=(0,(4,2)), lw=1.2, zorder=3)
ax.plot(range(5), rec, "o", color="#FF6B4A", ms=4, mec="white", mew=0.4, zorder=4)
ax.plot(range(5), it, color="#C000C0", lw=1.2, zorder=3)
ax.plot(range(5), it, "s", color="#C000C0", ms=4, mec="white", mew=0.4, zorder=4)
# drop1/2/3 markers at their tail points (rec = circle, iter = square, smaller)
for scen, off in [("drop3", 0.25), ("drop2", 0.5), ("drop1", 0.75)]:
    r2, ie2 = load("maize", scen)
    xr = r2["Time"].values; rr = r2["Mean"].values[-len(xr):] if len(xr)>=len(r2["Mean"].values) else r2["Mean"].values
    # find indices in 43..47
    for t, v in zip(r2["Time"].values[-3:], r2["Mean"].values[-3:]):
        j = list(x).index(t) if t in list(x) else None
        if j is not None:
            ax.plot([j+off], [v], "o", color="#FFB6A6", ms=3, mec="white", mew=0.3, zorder=4)
    for t, v in zip(ie2["Time"].values, ie2["Mean"].values):
        j = list(x).index(t) if t in list(x) else None
        if j is not None:
            ax.plot([j-off], [v], "s", color="#E39AE3", ms=3, mec="white", mew=0.3, zorder=4)
ax.set_xticks(range(5)); ax.set_xticklabels([str(t) for t in x], fontsize=7)
ax.set_ylim(0, 0.55); ax.set_yticks([0,0.2,0.4]); ax.tick_params(labelsize=7)
ax.set_xlabel("DAS", fontsize=8); ax.set_ylabel("Missing-point PCC", fontsize=8)
ax.set_title("Maize (delete last block)", fontsize=8)
ax.spines[["top","right"]].set_visible(False)

# ---- arab: gap_mid (12-16) + drop_last3 (19-21), rec vs iter ----
ax = axes[1]
r, ie = load("arab", "gap_mid")
xg = r["Time"].values[-5:]  # 12..16
rec_g = r["Mean"].values[-5:]; it_g = ie["Mean"].values
r3, ie3 = load("arab", "drop_last3")
xt = r3["Time"].values[-3:]  # 19..21
rec_t = r3["Mean"].values[-3:]; it_t = ie3["Mean"].values

# combine on a single x axis: use position 0..4 for gap (12-16), 5..7 for drop_last (19-21)
xx = list(range(5)) + [5.5, 6.5, 7.5]
rec_all = list(rec_g) + list(rec_t)
it_all = list(it_g) + list(it_t)
ax.plot(range(5), rec_g, color="#FF6B4A", ls=(0,(4,2)), lw=1.2, zorder=3)
ax.plot(range(5), rec_g, "o", color="#FF6B4A", ms=4, mec="white", mew=0.4, zorder=4)
ax.plot(range(5), it_g, color="#C000C0", lw=1.2, zorder=3)
ax.plot(range(5), it_g, "s", color="#C000C0", ms=4, mec="white", mew=0.4, zorder=4)
ax.plot([5.5,6.5,7.5], rec_t, color="#FF6B4A", ls=(0,(4,2)), lw=1.2, zorder=3)
ax.plot([5.5,6.5,7.5], rec_t, "o", color="#FF6B4A", ms=4, mec="white", mew=0.4, zorder=4)
ax.plot([5.5,6.5,7.5], it_t, color="#C000C0", lw=1.2, zorder=3)
ax.plot([5.5,6.5,7.5], it_t, "s", color="#C000C0", ms=4, mec="white", mew=0.4, zorder=4)
ax.axvline(5.2, color="0.7", lw=0.5, ls=":")
ax.set_xticks([0,1,2,3,4,5.5,6.5,7.5])
ax.set_xticklabels(["12","13","14","15","16","19","20","21"], fontsize=7)
ax.set_ylim(0, 0.35); ax.set_yticks([0,0.1,0.2,0.3]); ax.tick_params(labelsize=7)
ax.set_xlabel("DAS", fontsize=8); ax.set_ylabel("Missing-point PCC", fontsize=8)
ax.set_title("Arabidopsis (middle gap + tail drop)", fontsize=8)
ax.spines[["top","right"]].set_visible(False)

handles = [Line2D([],[], color="#FF6B4A", lw=1.2, ls=(0,(4,2)), marker="o", ms=4, mec="white", mew=0.4, label="recursive"),
           Line2D([],[], color="#C000C0", lw=1.2, marker="s", ms=4, mec="white", mew=0.4, label="iterative (from last true)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5,0.0))
fig.tight_layout(rect=[0,0.06,1,1])
fig.savefig(OUT+"fig_missing_panel.svg"); fig.savefig(OUT+"fig_missing_panel.png", dpi=160)
plt.close(fig)
print("saved fig_missing_panel (per-timepoint)")
