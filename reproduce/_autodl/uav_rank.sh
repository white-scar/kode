#!/bin/bash
# UAV-RGB rank: 两个环境同时跑
#   drought   : 10 进程 (每进程 1 折)
#   irrigated :  5 进程 (每进程 2 折)
R=/root/autodl-tmp/qkn/reproduce
L=/root/autodl-tmp/qkn/outputs/reproduce
PY=/root/miniconda3/bin/python
cd $R || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
: > $L/_uav_rank.log
echo "[$(date +%H:%M:%S)] drought start (10 procs)" >> $L/_uav_rank.log
for f in 0 1 2 3 4 5 6 7 8 9; do
  setsid nohup $PY knode_rank_genetics_value_uav.py --env drought --folds $f --n_epochs 1000 \
    > $L/_uav_rank_drought_f${f}.log 2>&1 < /dev/null &
done
echo "[$(date +%H:%M:%S)] irrigated start (5 procs)" >> $L/_uav_rank.log
for pair in 0,1 2,3 4,5 6,7 8,9; do
  setsid nohup $PY knode_rank_genetics_value_uav.py --env irrigated --folds $pair --n_epochs 1000 \
    > $L/_uav_rank_irrigated_${pair/,/_}.log 2>&1 < /dev/null &
done
wait
for env in drought irrigated; do
  $PY knode_rank_genetics_value_uav.py --env $env --aggregate >> $L/_uav_rank.log 2>&1
  echo "[$(date +%H:%M:%S)] $env aggregated" >> $L/_uav_rank.log
done
echo "ALL DONE" >> $L/_uav_rank.log