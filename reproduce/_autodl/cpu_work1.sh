#!/bin/bash
cd /root/autodl-tmp/qkn/reproduce || exit 1
PY=/root/miniconda3/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
LOGF=/root/autodl-tmp/qkn/outputs/reproduce/_cpu_work1.log
: > $LOGF
if [ -f "/root/autodl-tmp/qkn/outputs/reproduce\knode_rank_genetics_value_rice_f1.csv" ]; then echo "[exists] knode_rank_genetics_value_rice.py f1" >> $LOGF; else
  echo "[$(date +%H:%M:%S)] START knode_rank_genetics_value_rice.py f1" >> $LOGF
  $PY knode_rank_genetics_value_rice.py --fold 1 --n_epochs 1000 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   knode_rank_genetics_value_rice.py f1" >> $LOGF
fi
if [ -f "/root/autodl-tmp/qkn/outputs/reproduce\knode_rank_genetics_value_uav_f7.csv" ]; then echo "[exists] knode_rank_genetics_value_uav.py f7" >> $LOGF; else
  echo "[$(date +%H:%M:%S)] START knode_rank_genetics_value_uav.py f7" >> $LOGF
  $PY knode_rank_genetics_value_uav.py --fold 7 --n_epochs 1000 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   knode_rank_genetics_value_uav.py f7" >> $LOGF
fi
if [ -f "/root/autodl-tmp/qkn/outputs/reproduce\rank_select_v2_skipaffine_uav_f3.csv" ]; then echo "[exists] knode_rank_select_v2_skipaffine_uav.py f3" >> $LOGF; else
  echo "[$(date +%H:%M:%S)] START knode_rank_select_v2_skipaffine_uav.py f3" >> $LOGF
  $PY knode_rank_select_v2_skipaffine_uav.py --fold 3 --n_epochs 1000 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   knode_rank_select_v2_skipaffine_uav.py f3" >> $LOGF
fi
echo DONE_CPU_WORK1 >> $LOGF
