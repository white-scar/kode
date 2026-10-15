# -*- coding: utf-8 -*-
"""_rr_innerens_watch_merge.py -- 等 arab/rice 的 rr inner-5 10 折跑完, 自动合并成正式表."""
import os, time, subprocess, glob
import pandas as pd
ROOT = "/data/fandata/dashuaige"
os.chdir(ROOT)

def running(pat):
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    return bool([p for p in r.stdout.split() if p.isdigit()])

print("watch start", flush=True)
while running("dynamicGP_RR_innerens_arab") or running("dynamicGP_RR_innerens_rice"):
    time.sleep(60)
print("all workers finished", flush=True)

for ds in ("arab", "rice"):
    base = f"outputs/dynamicgp_rr_innerens_{ds}_par"
    out  = f"outputs/dynamicgp_rr_innerens_{ds}"
    fs = sorted(glob.glob(os.path.join(base, "fold*", "snapshot_metrics_long.csv")))
    if not fs:
        print(ds, "no fold outputs!", flush=True); continue
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d.sort_values(["Method", "Iteration", "Trait", "Time"]).reset_index(drop=True)
    os.makedirs(out, exist_ok=True)
    d.to_csv(os.path.join(out, "snapshot_metrics_long.csv"), index=False)
    d.groupby(["Method", "Time"])[["PCC","MSE","NMSE","R2"]].mean().reset_index() \
     .to_csv(os.path.join(out, "snapshot_metrics_by_time.csv"), index=False)
    ov = d[d.Time != "t1"].groupby("Method")[["PCC","MSE","NMSE","R2"]].mean().reset_index()
    ov.to_csv(os.path.join(out, "snapshot_metrics_overall.csv"), index=False)
    print(ds, "merged:", len(fs), "folds,", len(d), "rows", flush=True)
    print(ov.round(6).to_string(index=False), flush=True)

open("outputs/dynamicgp_rr_innerens_DONE.txt", "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + " merged\n")
print("DONE", flush=True)