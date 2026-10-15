#!/bin/bash
# _hp_time.sh -- 图六 重跑前的计时冒烟: 每数据集 1 foldset x 50 epoch, 量出单 config 全量(10 foldset x 1000 ep)耗时
#   rc 单独一行取, 避免 $(date) 吃掉 $?
set -u
cd /root/autodl-tmp/qkn/reproduce || exit 9
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
LOGD=/root/autodl-tmp/qkn/outputs/reproduce/_hp_time_logs; mkdir -p "$LOGD"
: > "$LOGD/time.txt"
run_one() {
  local ds="$1" t0 t1 rc
  t0=$(date +%s)
  $PY knode_hyperparam_ablation.py --dataset "$ds" --_single base \
      --epochs 50 --foldsets 1 --drift affine --mode ode_hyper,iter_hyper \
      --tag _time50 > "$LOGD/${ds}_time50.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "$ds rc=$rc seconds=$((t1-t0))" >> "$LOGD/time.txt"
}
export -f run_one; export PY LOGD
( run_one arab ) &
( run_one uav ) &
wait
echo TIMING_DONE >> "$LOGD/time.txt"