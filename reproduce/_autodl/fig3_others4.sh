#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
R=/root/autodl-tmp/qkn/reproduce
PY=/root/miniconda3/bin/python
pkill -f "python knode_cond_ladder_" 2>/dev/null
sleep 4
rm -f $L/knode_cond_ladder_arab_*.csv $L/knode_cond_ladder_rice_*.csv $L/knode_cond_ladder_rgb_*.csv
cd $R || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
: > $L/_fig3_others.log
echo "[$(date '+%T')] ??(????): ?????? 2 ?" >> $L/_fig3_others.log
for ds in arab rice rgb; do
  setsid nohup $PY knode_cond_ladder_$ds.py --folds 0,1,2,3,4 --n_epochs 1000 > $L/_fig3_${ds}_A.log 2>&1 < /dev/null &
  echo "  $ds A folds=0-4  PID=$!" >> $L/_fig3_others.log
  setsid nohup $PY knode_cond_ladder_$ds.py --folds 5,6,7,8,9 --n_epochs 1000 > $L/_fig3_${ds}_B.log 2>&1 < /dev/null &
  echo "  $ds B folds=5-9  PID=$!" >> $L/_fig3_others.log
done
sleep 10
echo "  ?????: $(pgrep -c -f knode_cond_ladder)" >> $L/_fig3_others.log
while pgrep -f "knode_cond_ladder" > /dev/null; do sleep 30; done
for ds in arab rice rgb; do
  echo "[$(date '+%T')] ?? $ds" >> $L/_fig3_others.log
  $PY knode_cond_ladder_$ds.py --aggregate >> $L/_fig3_others.log 2>&1
done
echo "[$(date '+%T')] DONE_FIG3_OTHERS" >> $L/_fig3_others.log
