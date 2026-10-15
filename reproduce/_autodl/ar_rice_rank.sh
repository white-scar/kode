#!/bin/bash
# 拟南芥 + 水稻 rank 重跑（新口径）: 每数据集 5 进程 (每进程 2 折)
R=/root/autodl-tmp/qkn/reproduce
L=/root/autodl-tmp/qkn/outputs/reproduce
PY=/root/miniconda3/bin/python
cd $R || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
: > $L/_ar_rank.log
for ds in arab rice; do
  echo "[$(date +%H:%M:%S)] $ds start" >> $L/_ar_rank.log
  for pair in 0,1 2,3 4,5 6,7 8,9; do
    setsid nohup $PY knode_rank_genetics_value_$ds.py --folds $pair --n_epochs 1000 \
      > $L/_ar_rank_${ds}_${pair/,/_}.log 2>&1 < /dev/null &
  done
done
wait
for ds in arab rice; do
  echo "[$(date +%H:%M:%S)] aggregate $ds" >> $L/_ar_rank.log
  $PY knode_rank_genetics_value_$ds.py --aggregate >> $L/_ar_rank.log 2>&1
done
echo "AR_RICE_DONE" >> $L/_ar_rank.log