#!/bin/bash
# _maize_dg_rank_all.sh -- 玉米 MAGIC 的 D_G x rank 联合扫描 (Full Kode)
#   跑法: 每 (fold, rank) 一个进程 = 10 x 4 = 40 进程; 每进程 6 个 D_G x 5 inner = 30 个模型
#   每个 cell 单独落盘 dg_rank_maize/cells/f{fold}_dg{dg}_r{rank}_ep{epochs}.csv
#   cells 已存在的 cell 会被跳过 => 任何重跑都不会覆盖已有结果
#   先等 cond ladder 结束再开始 (避免同时 70 个进程挤爆 24G 显存)
set -u
cd /root/autodl-tmp/qkn || exit 9
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOGD=outputs/reproduce/_maize_dgrank_logs
mkdir -p "$LOGD"
: > _maize_dg_rank_all.log
echo "[$(date '+%F %T')] waiting for cond ladder to finish ..." >> _maize_dg_rank_all.log
for i in $(seq 1 240); do
  pgrep -f '[k]node_cond_ladder' > /dev/null || break
  sleep 15
done
echo "[$(date '+%F %T')] cond ladder done (or timeout), start maize dg_rank: 40 processes" >> _maize_dg_rank_all.log
for f in 0 1 2 3 4 5 6 7 8 9; do
  for r in 1 2 3 4; do
    nohup $PY reproduce/knode_maize_dg_rank_joint.py --run --folds $f --ranks $r > "$LOGD/f${f}_r${r}.log" 2>&1 &
  done
  echo "[$(date '+%F %T')] launched fold $f ranks 1-4" >> _maize_dg_rank_all.log
done
echo "[$(date '+%F %T')] 40 processes launched" >> _maize_dg_rank_all.log
wait
$PY reproduce/knode_maize_dg_rank_joint.py --aggregate-only > "$LOGD/agg.log" 2>&1
echo "[$(date '+%F %T')] aggregate rc=$?" >> _maize_dg_rank_all.log
date '+%F %T' > outputs/reproduce/_maize_dg_rank.DONE
echo "[$(date '+%F %T')] DONE" >> _maize_dg_rank_all.log
