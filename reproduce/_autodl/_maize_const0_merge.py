# -*- coding: utf-8 -*-
"""_maize_const0_merge.py -- 把玉米 const(全0) 补算合并进正式文件
默认只校验+打印对比; 加 --apply 才写回。
合并内容: (1) outputs/reproduce/knode_rank_genetics_value_f{0..9}.csv 的 const 行 + 重算 10fold_agg
          (2) pic-code/pic4/fig_rank_energy.csv 的 const 行 (panel c 那条灰线)
写回前统一备份到 outputs/reproduce/_bak_pre_const0_20260920/
"""
import sys, shutil
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent                    # reproduce/_autodl
REPRO = HERE.parent                                       # reproduce
ROOT = REPRO.parent                                       # 仓库根
RD = ROOT / "outputs" / "reproduce"
TMP = RD / "_maize_const0_tmp"
BAK = RD / "_bak_pre_const0_20260920"
ENERGY = ROOT / "reproduce" / "pic-code" / "pic4" / "fig_rank_energy.csv"
RANKS = [1, 2, 3, 4, 6, 8, 10, 16, 20, 30, 50]
APPLY = "--apply" in sys.argv
print("ROOT =", ROOT, "| APPLY =", APPLY)

# ---------- 0) 校验临时结果 ----------
bad = []
for f in range(10):
    p = TMP / ("knode_rank_genetics_value_f%d.csv" % f)
    if not p.is_file():
        bad.append(str(p)); continue
    d = pd.read_csv(p); dc = d[d.cond == "const"]
    if sorted(dc["rank"].tolist()) != RANKS:
        bad.append("%s const rank 不全" % p.name)
if bad:
    print("[未通过]"); [print("  ", b) for b in bad]; sys.exit(1)
print("[校验] 10 个 fold 的 const 行齐全 (每折 11 rank)")

# ---------- 1) const 档 pcc/mse 前后对比 ----------
rows = []
for f in range(10):
    n = pd.read_csv(TMP / ("knode_rank_genetics_value_f%d.csv" % f))
    n = n[n.cond == "const"].set_index("rank")
    o = pd.read_csv(RD / ("knode_rank_genetics_value_f%d.csv" % f))
    o = o[o.cond == "const"].set_index("rank")
    for r in RANKS:
        rows.append(dict(rank=r, pcc_old=o.loc[r, "pcc"], pcc_new=n.loc[r, "pcc"],
                         mse_old=o.loc[r, "mse"], mse_new=n.loc[r, "mse"]))
g = pd.DataFrame(rows).groupby("rank")[["pcc_old", "pcc_new", "mse_old", "mse_new"]].mean()
g["dpcc"] = g.pcc_new - g.pcc_old
print("\n[对比 1] const 档: 全1(旧) -> 全0(新), 10 折均值")
print(g.round(4).to_string())

# ---------- 2) panel c const 谱能量前后对比 ----------
en_new = []
for f in range(10):
    d = pd.read_csv(TMP / ("knode_rank_energy_f%d.csv" % f))
    en_new.append(d[d.cond == "const"])
en_new = pd.concat(en_new).groupby("k", as_index=False)["energy"].mean()
en_old = pd.read_csv(ENERGY)
en_old = en_old[en_old.cond == "const"][["k", "energy"]]
cmp2 = en_old.merge(en_new, on="k", suffixes=("_old", "_new"))
cmp2["d"] = cmp2.energy_new - cmp2.energy_old
print("\n[对比 2] panel c 的 const 谱能量: 全1(旧) -> 全0(新)")
print(cmp2.round(4).to_string(index=False))

if not APPLY:
    print("\n(只校验不合并; 加 --apply 才写回)")
    sys.exit(0)

# ---------- 3) 备份 ----------
BAK.mkdir(parents=True, exist_ok=True)
tgts = sorted(RD.glob("knode_rank_genetics_value_f[0-9].csv")) + \
       [RD / "knode_rank_genetics_value_10fold_agg.csv", ENERGY]
for p in tgts:
    if p.is_file() and not (BAK / p.name).exists():
        shutil.copy2(p, BAK / p.name)
print("\n[备份] ->", BAK, "文件数:", len(list(BAK.glob('*'))))

# ---------- 4) 合并 f-CSV (只换 const 行, 其余行原序保留) ----------
for f in range(10):
    p = RD / ("knode_rank_genetics_value_f%d.csv" % f)
    old = pd.read_csv(p)
    nonconst = old[old.cond != "const"]
    newc = pd.read_csv(TMP / ("knode_rank_genetics_value_f%d.csv" % f))
    newc = newc[newc.cond == "const"]
    assert list(nonconst.columns) == list(newc.columns)
    pd.concat([nonconst, newc], ignore_index=True).to_csv(p, index=False)

# ---------- 5) 校验: 非 const 行逐格未变 ----------
bad = 0
for f in range(10):
    a = pd.read_csv(BAK / ("knode_rank_genetics_value_f%d.csv" % f))
    b = pd.read_csv(RD / ("knode_rank_genetics_value_f%d.csv" % f))
    a = a[a.cond != "const"].sort_values(["cond", "rank"]).reset_index(drop=True)
    b = b[b.cond != "const"].sort_values(["cond", "rank"]).reset_index(drop=True)
    if not a.equals(b):
        bad += 1; print("  [不一致] fold", f)
print("[校验] 合并后非 const 行与备份逐格一致:", "OK" if bad == 0 else "%d 折不一致" % bad)
assert bad == 0

# ---------- 6) 重算 10fold_agg ----------
sys.path.insert(0, str(REPRO))
import os, importlib
os.chdir(str(REPRO))        # 原脚本的 OUT=Path('../outputs/reproduce') 是相对 CWD 的, 必须切到 reproduce/
R = importlib.import_module("knode_rank_genetics_value")
R.aggregate()
agg = pd.read_csv(RD / "knode_rank_genetics_value_10fold_agg.csv")
print("\n[新 agg] const 档:")
print(agg[agg.cond == "const"][["cond", "rank", "pcc_mean", "pcc_std", "mse_mean"]].round(4).to_string(index=False))

# ---------- 7) 更新 panel c 的 const 行 ----------
en = pd.read_csv(ENERGY)
keep = en[en.cond != "const"]
new_rows = pd.DataFrame(dict(cond="const", k=en_new["k"].values, energy=en_new["energy"].values))
merged = pd.concat([keep, new_rows], ignore_index=True)
merged.to_csv(ENERGY, index=False)
print("\n[panel c] 已更新", ENERGY, "的 const 行; 行数", len(merged))
print("\nDONE; 备份在", BAK)