#!/bin/sh
cd /data/fandata/dashuaige/reproduce
PY=/data/fandata/dashuaige/miniconda3/envs/database/bin/python
OUTD=/data/fandata/dashuaige/outputs/reproduce
CUDA_VISIBLE_DEVICES=1 nohup $PY knode_dynamic_baselines_full_rice.py --seeds 42 --iters 0,1,2 --tag _p1 > $OUTD/rice_baselines_p1.log 2>&1 &
echo P1_PID=$!
CUDA_VISIBLE_DEVICES=2 nohup $PY knode_dynamic_baselines_full_rice.py --seeds 42 --iters 3,4,5 --tag _p2 > $OUTD/rice_baselines_p2.log 2>&1 &
echo P2_PID=$!
CUDA_VISIBLE_DEVICES=3 nohup $PY knode_dynamic_baselines_full_rice.py --seeds 42 --iters 6,7 --tag _p3 > $OUTD/rice_baselines_p3.log 2>&1 &
echo P3_PID=$!
CUDA_VISIBLE_DEVICES=4 nohup $PY knode_dynamic_baselines_full_rice.py --seeds 42 --iters 8,9 --tag _p4 > $OUTD/rice_baselines_p4.log 2>&1 &
echo P4_PID=$!
sleep 12
pgrep -af knode_dynamic_baselines_full_rice.py | grep -v pgrep
echo "--- logs ---"
ls -la $OUTD/rice_baselines_p*.log
