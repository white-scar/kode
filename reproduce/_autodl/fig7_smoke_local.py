# -*- coding: utf-8 -*-
"""图七四份冒烟: maize/arab (arab 脚本) + rice (rice 脚本) + uav (rgb 脚本), 全部 --smoke"""
import subprocess, sys, time
from pathlib import Path
ROOT = Path(r"D:\pythonproject\knode")
PY = r"D:\APP\ANACONDA\python.exe"
jobs = [
    ("maize", ROOT / "reproduce" / "arab" / "knode_extrap_maize_arab_v2.py", ["--dataset", "maize", "--smoke"]),
    ("arab",  ROOT / "reproduce" / "arab" / "knode_extrap_maize_arab_v2.py", ["--dataset", "arab", "--smoke"]),
    ("rice",  ROOT / "reproduce" / "rice" / "knode_extrap_maize_rice_v2.py", ["--dataset", "rice", "--smoke"]),
    ("rgb",   ROOT / "reproduce" / "knode_extrap_maize_rgb_v2.py",           ["--dataset", "rgb", "--smoke"]),
]
logs = ROOT / "_fig7_smoke_logs"; logs.mkdir(exist_ok=True)
procs = {}
for tag, script, args in jobs:
    f = open(logs / (tag + ".log"), "w", encoding="utf-8")
    p = subprocess.Popen([PY, str(script)] + args, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    procs[tag] = (p, f, time.time())
    print("start", tag, "pid", p.pid, flush=True)
while procs:
    for tag in list(procs):
        p, f, t0 = procs[tag]
        if p.poll() is not None:
            f.close()
            print("END %-6s rc=%d  %.1f s" % (tag, p.returncode, time.time() - t0), flush=True)
            del procs[tag]
    time.sleep(5)
print("ALL SMOKE DONE")
