#!/bin/bash
# _maize_condladder_run.sh -- 玉米 cond ladder 首次生成 (5 档 x 10 fold x 5 inner x 1000 ep)
#   用临时脚本 _tmp_maize_condladder.py, 产物另存:
#     outputs/reproduce/_maize_condladder_tmp/cond_ladder_maize_dg50_r2/{f0..f9.csv, trait_f*.csv, agg.csv, *_meta.json}
#   绝不触碰正式 cond_ladder_maize_dg50_r2/ (合并另行确认后做)
set -u
cd /root/autodl-tmp/qkn/reproduce || exit 9
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
LOGD=/root/autodl-tmp/qkn/outputs/reproduce/_maize_condladder_logs; mkdir -p "$LOGD"
LOG=/root/autodl-tmp/qkn/_maize_condladder.log
: > $LOG
echo "[$(date '+%F %T')] start maize cond ladder: 10 folds parallel, 5 conds x 5 inner x 1000 ep" >> $LOG
run_fold() {
  local f="$1"
  $PY _tmp_maize_condladder.py --folds "$f" --n_epochs 1000 > "$LOGD/f${f}.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] fold $f rc=$rc" >> $LOG
}
export -f run_fold; export PY LOGD LOG
for f in 0 1 2 3 4 5 6 7 8 9; do ( run_fold $f ) & done
wait
echo "[$(date '+%F %T')] folds done, aggregate ..." >> $LOG
$PY _tmp_maize_condladder.py --aggregate >> $LOGD/aggregate.log 2>&1
echo "[$(date '+%F %T')] aggregate rc=$?" >> $LOG
date '+%F %T' > /root/autodl-tmp/qkn/outputs/reproduce/_maize_condladder.DONE
echo "[$(date '+%F %T')] ALL DONE" >> $LOG