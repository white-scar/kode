# -*- coding: utf-8 -*-
"""_ststp_single_batch.py - 玉米 MAGIC 50 个性状逐个训练 ST-STP P=1 模型（支持断点续跑）

调用上一轮同构脚本 knode_ststp_single_maize.py（= knode_tp1only_dynamic.py 的副本，仅限 ststp）
用法:
  cd <root>/reproduce && python _ststp_single_batch.py                 # 全量 50 性状
  cd <root>/reproduce && python _ststp_single_batch.py --smoke --only 1  # 冒烟（写 _smoke 目录）
"""
import argparse, os, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

ROOT = Path(os.environ.get("KNODE_ROOT", "..")).resolve()
REP = ROOT / "reproduce"
DS_CFG = {
    "maize": ("knode_ststp_single_maize.py", "knode_ststp_single_maize",
              ROOT / "data" / "maize_magic" / "maize_trait_BLUP_50_NArmv_traits_afterClustering.csv"),
    "rice":  ("knode_ststp_single_rice.py", "knode_ststp_single_rice",
              ROOT / "data" / "rice_salinity" / "rice_indica_control_traits.csv"),
    "arab":  ("knode_ststp_single_arab.py", "knode_ststp_single_arab",
              ROOT / "data" / "arabidopsis" / "BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv"),
}
SCRIPT = BASE_OUT = DATA = None
META = ["bio_ID", "accession", "DAS"]
NEED = ["metrics_by_trait_time.csv", "metrics_by_trait_time_per_iter.csv",
        "metrics_by_trait_time_std.csv", "metrics_overall.csv",
        "predictions_long.csv.gz", "run_config.json", "train.log"]

def trait_list():
    d = pd.read_csv(DATA, nrows=1)
    return [c for c in d.columns if c not in META and not c.startswith("Unnamed") and c != ""]

def slug(t, n=40):
    s = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in t)
    return s[:n].rstrip("._-") or "t"

OUT = BASE_OUT  # 由 main 按 smoke 覆盖

def outdir(idx, t):
    return OUT / ("single_p%02d_%s" % (idx + 1, slug(t)))

def complete(d):
    return all((d / f).exists() for f in NEED)

def run_one(args):
    idx, t, epochs = args
    d = outdir(idx, t)
    if complete(d):
        return (idx, t, "skip", 0.0)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(d / "_run.log", "w", encoding="utf-8") as fh:
        rc = subprocess.call([sys.executable, str(SCRIPT), "--mode", "single", "--traits", t,
                              "--fams", "ststp", "--iters", "0,1,2", "--epochs", str(epochs),
                              "--outdir", str(d)], cwd=str(REP), stdout=fh, stderr=subprocess.STDOUT)
    return (idx, t, "ok" if rc == 0 else ("rc=%d" % rc), time.time() - t0)

def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default="", help="只跑第几个性状(1-based, 逗号分隔)")
    ap.add_argument("--dataset", default="maize", choices=["maize", "rice", "arab"])
    a = ap.parse_args()
    global SCRIPT, BASE_OUT, DATA
    _sf, _on, DATA = DS_CFG[a.dataset]
    SCRIPT = REP / _sf
    BASE_OUT = ROOT / "outputs" / "reproduce" / _on

    OUT = BASE_OUT / "_smoke" if a.smoke else BASE_OUT
    OUT.mkdir(parents=True, exist_ok=True)
    LOG = OUT / "batch.log"
    ts = trait_list()
    idxs = [int(x) - 1 for x in a.only.split(",") if x.strip()] if a.only else list(range(len(ts)))
    epochs = min(a.epochs, 20) if a.smoke else a.epochs
    par = 1 if a.smoke else max(1, a.parallel)

    def log(m):
        line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), m)
        print(line, flush=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log("=== batch start: %d traits, iters=0,1,2, epochs=%d, parallel=%d, outdir=%s ==="
        % (len(idxs), epochs, par, OUT))
    jobs = [(i, ts[i], epochs) for i in idxs]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=par) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for f in as_completed(futs):
            i, t, st, dt = f.result()
            log("  p%02d %-60s %-6s %6.1fs" % (i + 1, t[:60], st, dt))
    ok = sum(1 for i in idxs if complete(outdir(i, ts[i])))
    log("=== batch done: %d/%d 完整, 用时 %.1f min ===" % (ok, len(idxs), (time.time() - t0) / 60))

if __name__ == "__main__":
    main()
