#!/bin/bash
R=/root/autodl-tmp/qkn/reproduce
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/qkn/outputs/reproduce/_rice_abl.log
cd $R || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
: > $LOG
echo "[$(date +%H:%M:%S)] START rice affine (10 folds)" >> $LOG
$PY knode_rice_np_tp1_affine.py --foldsets 10 --epochs 1000 --drift affine --mode ode_hyper,iter_hyper --tag _affine >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] END rice affine" >> $LOG
$PY knode_rice_np_tp1_affine.py --foldsets 10 --epochs 1000 --drift linear --mode ode_hyper,iter_hyper --tag _linear >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] END rice linear" >> $LOG
echo "DONE" >> $LOG