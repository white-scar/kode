#!/bin/sh
cd /data/fandata/dashuaige/reproduce
PY=/data/fandata/dashuaige/miniconda3/envs/database/bin/python
OUTD=/data/fandata/dashuaige/outputs/reproduce
CUDA_VISIBLE_DEVICES=1 nohup $PY knode_maize_np_tp1_skip_affine.py --foldsets 10 --epochs 1000 --drift linear --mode ode_hyper,ode_hyper_skip --tag _linear > $OUTD/maize_abl_l_rec.log 2>&1 &
echo L_REC_PID=$!
CUDA_VISIBLE_DEVICES=2 nohup $PY knode_maize_np_tp1_skip_affine.py --foldsets 10 --epochs 1000 --drift linear --mode iter_hyper,iter_hyper_skip --tag _linear > $OUTD/maize_abl_l_iter.log 2>&1 &
echo L_ITER_PID=$!
CUDA_VISIBLE_DEVICES=3 nohup $PY knode_maize_np_tp1_skip_affine.py --foldsets 10 --epochs 1000 --drift affine --mode ode_hyper,ode_hyper_skip --tag _affine > $OUTD/maize_abl_a_rec.log 2>&1 &
echo A_REC_PID=$!
CUDA_VISIBLE_DEVICES=4 nohup $PY knode_maize_np_tp1_skip_affine.py --foldsets 10 --epochs 1000 --drift affine --mode iter_hyper,iter_hyper_skip --tag _affine > $OUTD/maize_abl_a_iter.log 2>&1 &
echo A_ITER_PID=$!
sleep 20
pgrep -af knode_maize_np_tp1_skip_affine.py | grep -v pgrep
head -3 $OUTD/maize_abl_a_rec.log
