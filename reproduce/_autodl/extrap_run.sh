#!/bin/bash
# ?7 ?? ? ? skip ??? knode_extrap_maize_arab_v2.py ???: n_test=5, 10 ?, 800 epoch?
cd /root/autodl-tmp/qkn/reproduce || exit 1
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOGD=/root/autodl-tmp/qkn/outputs/reproduce
rm -f $LOGD/knode_extrap_maize_arab/*_smoke*
for DS in maize arab; do
  case $DS in
    maize) DIMS=50 ;; arab) DIMS=30 ;;
  esac
  LOG=$LOGD/_extrap_${DS}_noskip.log
  : > $LOG
  echo "[$(date '+%m-%d %H:%M:%S')] START $DS (no-skip, n_test=5, 10 foldsets, 800 epochs)" >> $LOG
  $PY knode_extrap_maize_arab_v2.py --dataset $DS --n_test 5 --foldsets 1,2,3,4,5,6,7,8,9,10 --epochs 800 >> $LOG 2>&1
  echo "[$(date '+%m-%d %H:%M:%S')] END   $DS rc=$?" >> $LOG
done
echo DONE_EXTRAP >> $LOGD/_extrap_maize_noskip.log
