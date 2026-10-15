#!/bin/bash
# _fig5_baselines_all.sh -- 图五 基线侧重跑 (arab / rice / uav-drought), 新口径
#   arab dim=30 / rice dim=10 (DAS30-42) / uav dim=50 ; 5 个基线族 x 10 iter x 4 seed x 1000ep
#   --tag _v2 => 输出 knode_dynamic_baselines_full_{arab,rice,uav}_v2.* (不覆盖旧文件)
#   只开 3 个进程 (每数据集一个), 与图四并行, 显存占用很小
set -u
cd /root/autodl-tmp/qkn/reproduce || exit 9
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
LOGD=/root/autodl-tmp/qkn/outputs/reproduce/_fig5_logs; mkdir -p "$LOGD"
LOG=/root/autodl-tmp/qkn/_fig5_baselines_all.log
: > $LOG
COMMON="--iters 0,1,2,3,4,5,6,7,8,9 --seeds 42,1234,2024,2025 --epochs 1000 --tag _v2"
run_one() {  # $1=tag $2=script $3=dim
  echo "[$(date '+%F %T')] START $1 (dim=$3)" >> $LOG
  $PY "$2" $COMMON --grm_dim "$3" > "$LOGD/$1.log" 2>&1
  echo "[$(date '+%F %T')] END   $1 rc=$?" >> $LOG
}
export -f run_one; export PY LOG COMMON
( run_one arab arab/knode_dynamic_baselines_full_arab.py 30 ) &
( run_one rice rice/knode_dynamic_baselines_full_rice.py 10 ) &
( run_one uav  maize_RGB/knode_dynamic_baselines_full_uav.py 50 ) &
wait
date '+%F %T' > /root/autodl-tmp/qkn/outputs/reproduce/_fig5_baselines.DONE
echo "[$(date '+%F %T')] ALL DONE" >> $LOG
