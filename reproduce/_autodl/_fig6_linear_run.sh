#!/bin/bash
# _fig6_linear_run.sh -- 图六 component 面板缺的 linear drift 档 (affine 已由今天 hp 的 _hp_base 提供)
#   3 个数据集并行; tag=_hp_base_linear (绝不覆盖 affine 的 _hp_base)
set -u
cd /root/autodl-tmp/qkn/reproduce || exit 9
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
LOGD=/root/autodl-tmp/qkn/outputs/reproduce/_fig6_linear_logs; mkdir -p "$LOGD"
LOG=/root/autodl-tmp/qkn/_fig6_linear.log
: > $LOG
run_ds() {
  local ds="$1"
  echo "[$(date '+%F %T')] START $ds (drift=linear)" >> $LOG
  $PY knode_hyperparam_ablation.py --dataset "$ds" --_single base \
      --epochs 1000 --foldsets 10 --drift linear --mode ode_hyper,iter_hyper \
      --tag _hp_base_linear > "$LOGD/${ds}.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] END   $ds rc=$rc" >> $LOG
}
export -f run_ds; export PY LOGD LOG
( run_ds arab ) &
( run_ds rice ) &
( run_ds uav ) &
wait
date '+%F %T' > /root/autodl-tmp/qkn/outputs/reproduce/_fig6_linear.DONE
echo "[$(date '+%F %T')] ALL DONE" >> $LOG