# -*- coding: utf-8 -*-
"""_local_uav_chain.py - 在【本机 GPU】上顺序跑 UAV TP1-only 的 E1/E2/E3

顺序: E1(all, 19 性状) -> E2(single x3: TGI/GCC/RGBVI) -> E3(few, P=5)
输出: outputs/reproduce/knode_tp1only_uav/{all,single_*,few5}  (每目录 7 件产物)
日志: outputs/reproduce/knode_tp1only_uav/local_chain.log + 每个任务单独 .log
"""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reproduce"
SCRIPT = REP / "maize_RGB" / "knode_tp1only_dynamic_uav.py"
OUT = ROOT / "outputs" / "reproduce" / "knode_tp1only_uav"
LOG = OUT / "local_chain.log"
OUT.mkdir(parents=True, exist_ok=True)

ENV = dict(os.environ)
ENV["OMP_NUM_THREADS"] = "6"; ENV["MKL_NUM_THREADS"] = "6"
ENV["OPENBLAS_NUM_THREADS"] = "6"; ENV["NUMEXPR_NUM_THREADS"] = "6"

def log(m):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write("[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), m))

JOBS = [
    ("E1_all",   ["--mode", "all",  "--outdir", str(OUT / "all")]),
    ("E2_TGI",   ["--mode", "single", "--traits", "TGI",   "--outdir", str(OUT / "single_TGI")]),
    ("E2_GCC",   ["--mode", "single", "--traits", "GCC",   "--outdir", str(OUT / "single_GCC")]),
    ("E2_RGBVI", ["--mode", "single", "--traits", "RGBVI", "--outdir", str(OUT / "single_RGBVI")]),
    ("E3_few5",  ["--mode", "few", "--traits", "GCC,TGI,RGBVI,VARI,NRMBI", "--outdir", str(OUT / "few5")]),
]

log("=== LOCAL UAV CHAIN START (pid=%d) ===" % os.getpid())
for name, args in JOBS:
    log("start %s" % name)
    cmd = [sys.executable, str(SCRIPT)] + args + ["--iters", "0,1,2", "--epochs", "1000"]
    with open(OUT / ("run_%s.log" % name), "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=str(REP), stdout=fh, stderr=subprocess.STDOUT, env=ENV)
    log("end %s rc=%d" % (name, rc))
log("=== LOCAL UAV CHAIN ALL DONE ===")
