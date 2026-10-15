#!/bin/bash
# _maize_const0_run.sh -- 玉米 const 档 (全 0 输入) 补算: 10 个 fold 并行
#   只跑 const 档, 临时脚本 _tmp_maize_const0.py, 产物另存:
#     outputs/reproduce/_maize_const0_tmp/knode_rank_genetics_value_f{0..9}.csv
#     outputs/reproduce/_maize_const0_tmp/uv3_const0/gen_const_r*_f*_i*.npz
#   绝不触碰正式的 knode_rank_genetics_value_f*.csv / _10fold_agg.csv (合并另行确认后做)
set -u
cd /root/autodl-tmp/qkn/reproduce || exit 9
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
LOGD=/root/autodl-tmp/qkn/outputs/reproduce/_maize_const0_logs; mkdir -p "$LOGD"
LOG=/root/autodl-tmp/qkn/_maize_const0.log
: > $LOG
echo "[$(date '+%F %T')] start maize const0: 10 folds parallel, 11 ranks x 5 inner x 1000 ep" >> $LOG
run_fold() {
  local f="$1"
  $PY _tmp_maize_const0.py --folds "$f" --n_epochs 1000 > "$LOGD/f${f}.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] fold $f rc=$rc" >> $LOG
}
export -f run_fold; export PY LOGD LOG
for f in 0 1 2 3 4 5 6 7 8 9; do ( run_fold $f ) & done
wait
date '+%F %T' > /root/autodl-tmp/qkn/outputs/reproduce/_maize_const0.DONE
echo "[$(date '+%F %T')] ALL DONE" >> $LOG