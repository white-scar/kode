# -*- coding: utf-8 -*-
"""服务器端: 修四份 cond ladder 的 run_meta.json folds 字段 (元数据修正)"""
import json
from pathlib import Path
R = Path("/root/autodl-tmp/qkn/outputs/reproduce")
for d in sorted(R.glob("cond_ladder_*")):
    p = d / "run_meta.json"
    if not p.is_file():
        continue
    j = json.loads(p.read_text(encoding="utf-8"))
    old = j.get("folds")
    j["folds"] = list(range(10))
    j["folds_note"] = "parallel workflow: rewritten by --aggregate from f*.csv (fixed 2026-09-20)"
    p.write_text(json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  fixed %-32s folds %s -> %s" % (d.name, old, j["folds"]))