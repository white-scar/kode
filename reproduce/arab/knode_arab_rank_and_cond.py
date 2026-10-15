#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""knode_arab_rank_and_cond.py -- 拟南芥: 一次跑完「rank 分析」+「cond 输入阶梯」两套数据。

为什么要合并:
    arab\knode_cond_ladder_arab.py:11 本来就是 `import knode_rank_genetics_value_arab as M`,
    即 cond ladder 把 rank 脚本当数据+模型模块用。二者共用同一份数据、同一个模型族、
    同一套 inner-5/十折协议, 所以放一个 driver 里跑最省事, 也不会有口径漂移。

本脚本只做三件事, 不重实现任何模型/指标:
    1) 修路径: 把 reproduce\ 与 arab\ 加进 sys.path; 把两个模块的 BASE/OUT/OUT_DIR/UV2
       在运行时重指到 knode 根 (它们被搬进子目录后 BASE=parent.parent 会指错一层)。
       **不修改 knode_rank_genetics_value_arab.py / knode_cond_ladder_arab.py 任何字节。**
    2) 依次跑: rank 分析(rank 扫描 x 5 条件) -> 聚合 -> cond ladder(5 条件) -> 聚合
    3) 跑完自检: 图三、图四需要哪些文件、是否齐全、各多少行

产出 (outputs/reproduce/):
    图四用: knode_rank_genetics_value_arab_f0..f9.csv + knode_rank_genetics_value_arab_10fold_agg.csv
    图三用: knode_cond_ladder_arab_f0..f9.csv + knode_cond_ladder_arab_trait_f0..f9.csv + knode_cond_ladder_arab_agg.csv
    (图三还要主方法的 knode_arab_tp1_ode_hyper_per_trait.csv 作 Full 参照, 由 knode_arab_np_tp1_affine.py 产出)

用法:
    python knode_arab_rank_and_cond.py                 # 只自检, 不跑
    python knode_arab_rank_and_cond.py --run           # 全跑 (10 fold, 1000 ep)
    python knode_arab_rank_and_cond.py --run --folds 0 # 冒烟: 只跑 fold0
    python knode_arab_rank_and_cond.py --run --dg 20   # 指定 D_G (默认用脚本里的 30)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # reproduce/arab/
REPRO = HERE.parent                             # reproduce/

# ---------------- 位置无关: 向上找同时有 data/ 与 outputs/ 的那一级 ----------------
def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").is_dir() and (p / "outputs").is_dir():
            return p
    raise SystemExit("找不到 knode 根目录（需要同时存在 data/ 与 outputs/）")


ROOT = find_root(HERE)
OUT = ROOT / "outputs" / "reproduce"

# import 解析: 两个模块自己 import 的东西 (knode_perfect_vs_predicted_operators 等) 都在 reproduce\
sys.path.insert(0, str(REPRO))
sys.path.insert(0, str(HERE))

import knode_rank_genetics_value_arab as RK      # noqa: E402
import knode_cond_ladder_arab as CL              # noqa: E402


def patch_paths(dg: int):
    """把两个模块的路径与维度常量在运行时改对 (不落盘、不改源文件)."""
    if dg:
        RK.GRM_DIM = dg
        CL.GRM_DIM = dg
    RK.BASE = ROOT                               # 原: parent.parent -> 搬进 arab\ 后指错
    RK.OUT = OUT
    RK.OUT.mkdir(parents=True, exist_ok=True)
    RK.UV2 = OUT / "uv2_arab"                    # rank 的 npz 缓存目录
    RK.UV2.mkdir(exist_ok=True)
    CL.OUT_DIR = OUT                             # 原: Path('../outputs/reproduce') CWD 相对
    CL.OUT_DIR.mkdir(parents=True, exist_ok=True)
    CL.M = RK                                    # 明确 cond ladder 用到的就是同一个已修好的模块
    print(f"[paths] ROOT={ROOT}")
    print(f"[paths] rank OUT={RK.OUT}  UV2={RK.UV2}")
    print(f"[paths] cond OUT_DIR={CL.OUT_DIR}")
    RK_RANK = getattr(RK, "RANK", 2)
    print(f"[proto] GRM_DIM={RK.GRM_DIM}  rank 网格={RK.RANKS}  CONDS={RK.CONDS}")
    print(f"[proto] cond ladder CONDS={CL.CONDS}  RANK={getattr(CL, chr(82)+chr(65)+chr(78)+chr(75), 2)}")


def report():
    need = {
        "图四 rank / 每折": [f"knode_rank_genetics_value_arab_f{i}.csv" for i in range(10)],
        "图四 rank / 聚合": ["knode_rank_genetics_value_arab_10fold_agg.csv"],
        "图三 cond / 每折": [f"knode_cond_ladder_arab_f{i}.csv" for i in range(10)],
        "图三 cond / 逐性状": [f"knode_cond_ladder_arab_trait_f{i}.csv" for i in range(10)],
        "图三 cond / 聚合": ["knode_cond_ladder_arab_agg.csv"],
        "图三 Full 参照(主方法)": ["knode_arab_tp1_ode_hyper_per_trait.csv"],
    }
    print()
    print("=" * 92)
    print("图三/图四 数据自检  (outputs/reproduce/ 与 outputs/kode_result/arab/)")
    print("=" * 92)
    ok_all = True
    for group, files in need.items():
        n_ok = 0
        rows = []
        for f in files:
            p = OUT / f
            if not p.exists():
                p = ROOT / "outputs" / "kode_result" / "arab" / f
            if p.exists():
                n_ok += 1
                try:
                    rows.append(sum(1 for _ in open(p, encoding="utf-8", errors="replace")) - 1)
                except Exception:
                    rows.append("?")
        flag = "OK " if n_ok == len(files) else "缺 "
        if n_ok != len(files):
            ok_all = False
        print(f"  [{flag}] {group:<24} {n_ok}/{len(files)} 个" +
              (f"   行数={rows[:3]}{'...' if len(rows) > 3 else ''}" if rows else ""))
    print()
    print("  下一步: pic-code\\pic4 出图四, pic-code\\pic3\\arab 出图三" if ok_all
          else "  还缺文件 —— 先 --run 跑完再来自检")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="真正执行(默认只做自检)")
    ap.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--dg", type=int, default=None, help="GRM PCA 维度; 不给则用模块自带默认(30)")
    ap.add_argument("--ranks", default=None, help="覆盖 rank 网格, 逗号分隔")
    ap.add_argument("--only-cond", action="store_true", help="只跑 cond ladder")
    ap.add_argument("--only-rank", action="store_true", help="只跑 rank 扫描")
    a = ap.parse_args()

    folds = [int(x) for x in a.folds.split(",") if x.strip()]
    patch_paths(a.dg)

    if not a.run:
        print("\n[dry] 未执行训练 (加 --run 才跑)。当前只做数据自检:")
        report()
        return

    if not a.only_cond:
        ranks = [int(x) for x in a.ranks.split(",")] if a.ranks else RK.RANKS
        print(f"\n########## 1/2 rank 分析  folds={folds} ranks={ranks} conds={RK.CONDS} epochs={a.epochs}")
        for f in folds:
            RK.run_fold(f, ranks, RK.CONDS, a.epochs)
        RK.aggregate()

    if not a.only_rank:
        print(f"\n########## 2/2 cond ladder  folds={folds} conds={CL.CONDS} epochs={a.epochs}")
        for f in folds:
            CL.run_fold(f, a.epochs)
        CL.aggregate()

    report()


if __name__ == "__main__":
    main()