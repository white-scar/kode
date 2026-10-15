#!/bin/bash
cd /root/autodl-tmp/qkn/reproduce || exit 1
PY=/root/miniconda3/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
LOGF=/root/autodl-tmp/qkn/outputs/reproduce/_cpu_work10.log
: > $LOGF
if [ -f "/root/autodl-tmp/qkn/outputs/reproduce\knode_rank_genetics_value_uav_f0.csv" ]; then echo "[exists] knode_rank_genetics_value_uav.py f0" >> $LOGF; else
  echo "[$(date +%H:%M:%S)] START knode_rank_genetics_value_uav.py f0" >> $LOGF
  $PY knode_rank_genetics_value_uav.py --fold 0 --n_epochs 1000 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   knode_rank_genetics_value_uav.py f0" >> $LOGF
fi
if [ -f "/root/autodl-tmp/qkn/outputs/reproduce\rank_select_v2_skipaffine_rice_f6.csv" ]; then echo "[exists] knode_rank_select_v2_skipaffine_rice.py f6" >> $LOGF; else
  echo "[$(date +%H:%M:%S)] START knode_rank_select_v2_skipaffine_rice.py f6" >> $LOGF
  $PY knode_rank_select_v2_skipaffine_rice.py --fold 6 --n_epochs 1000 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   knode_rank_select_v2_skipaffine_rice.py f6" >> $LOGF
fi
echo DONE_CPU_WORK10 >> $LOGF
