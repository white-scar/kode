#!/bin/bash
# _fig7_extrap_all.sh -- Fig.7 (外推 / forecasting) 主方法 Kode (no-skip + affine drift, rank=2)
#   四份, 每份 = 10 foldset x 5 inner x 1000 epoch (fold1-5 train / fold6 test, checkpoint selection 无早停):
#     maize  MAGIC     nt5  dim50  (DAS 15-47, P=50, N=330)
#     arab              nt2  dim30  (DAS  8-21, P=45, N=382)
#     rice   control    nt2  dim10  (DAS 30-42, P=20, N=226; 表型只用 DAS>=30)
#     uav    RGB drought nt2 dim50  (DAS 42-99, P=19, N=509)
#   本地 --smoke (1 fold x 50 ep) 四份全绿; maize/arab 与清 skip 前逐位一致 (max|diff|=0e0)
#   输出: outputs/reproduce/knode_extrap_<ds>_nt<n>/knode_extrap_*_v2_f10_nt<n>*.csv (+ _overall/_by_time/_by_trait/_summary/_meta)
#   排队: 等 Fig.6 (hp ablation) 跑完; 完成写 outputs/reproduce/_fig7_extrap.DONE
set -u
cd /root/autodl-tmp/qkn/reproduce || exit 9
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
LOGD=/root/autodl-tmp/qkn/outputs/reproduce/_fig7_logs; mkdir -p "$LOGD"
LOG=/root/autodl-tmp/qkn/_fig7_extrap_all.log
: > $LOG
echo "[$(date '+%F %T')] waiting for Fig.6 hp ablation to finish ..." >> $LOG
for i in $(seq 1 2880); do [ -f /root/autodl-tmp/qkn/outputs/reproduce/_hp_ablation.DONE ] && break; sleep 30; done
echo "[$(date '+%F %T')] start Fig.7 extrap (4 datasets) ..." >> $LOG

run_one() {  # $1=tag $2=script ; 其余参数透传
  local tag="$1" sc="$2"; shift 2
  echo "[$(date '+%F %T')] START $tag" >> $LOG
  $PY "$sc" "$@" > "$LOGD/$tag.log" 2>&1
  echo "[$(date '+%F %T')] END   $tag rc=$?" >> $LOG
}
export -f run_one; export PY LOGD LOG

( run_one maize arab/knode_extrap_maize_arab_v2.py --dataset maize --n_test 5 --epochs 1000 ) &
( run_one arab  arab/knode_extrap_maize_arab_v2.py --dataset arab  --n_test 2 --epochs 1000 ) &
( run_one rice  rice/knode_extrap_maize_rice_v2.py  --dataset rice  --n_test 2 --epochs 1000 ) &
( run_one uav   knode_extrap_maize_rgb_v2.py        --dataset rgb --env drought --n_test 2 --epochs 1000 ) &
wait
date '+%F %T' > /root/autodl-tmp/qkn/outputs/reproduce/_fig7_extrap.DONE
echo "[$(date '+%F %T')] ALL DONE" >> $LOG
