#!/bin/bash
# maize rank analysis, two passes:
#   pass1: (full, rank=2) over folds 0..9 sequentially -> model-creation order matches the main script
#   pass2: every other (cond, rank) combination, one fold per parallel job (no shared output file)
R=/root/autodl-tmp/qkn/reproduce
L=/root/autodl-tmp/qkn/outputs/reproduce
PY=/root/miniconda3/bin/python
cd $R || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
: > $L/_rank_launch.log
echo "[$(date '+%T')] pass1 start (full/r2, sequential)" >> $L/_rank_launch.log
$PY knode_rank_genetics_value.py --folds 0,1,2,3,4,5,6,7,8,9 --conds full --ranks 2 >> $L/_rank_pass1.log 2>&1
echo "[$(date '+%T')] pass1 done" >> $L/_rank_launch.log

for f in 0 1 2 3 4 5 6 7 8 9; do
  (
    $PY knode_rank_genetics_value.py --folds $f --conds full --ranks 1,3,4,6,8,10,16,20,30,50 > $L/_rank_f${f}_a.log 2>&1
    $PY knode_rank_genetics_value.py --folds $f --conds grmonly,tp1only,const,noise --ranks 1,2,3,4,6,8,10,16,20,30,50 > $L/_rank_f${f}_b.log 2>&1
  ) &
  echo "  fold $f PID=$!" >> $L/_rank_launch.log
done
wait
echo "[$(date '+%T')] all folds done, aggregating" >> $L/_rank_launch.log
$PY knode_rank_genetics_value.py --aggregate >> $L/_rank_launch.log 2>&1
echo "[$(date '+%T')] ALL DONE" >> $L/_rank_launch.log
