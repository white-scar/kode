#!/bin/bash
# ?3 ???? ? ? skip ? arab/rice/rgb ? 2 ???(5 ?/?) + ????
R=/root/autodl-tmp/qkn/reproduce
L=/root/autodl-tmp/qkn/outputs/reproduce
PY=/root/miniconda3/bin/python
cd $R || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# ??????
rm -f $L/knode_cond_ladder_arab_*.csv $L/knode_cond_ladder_rice_*.csv $L/knode_cond_ladder_rgb_*.csv

: > $L/_fig3_others.log
echo "[$(date '+%T')] ??: arab/rice/rgb ? 2 ? (folds 0-4 / 5-9), 1000 epoch, ? skip" >> $L/_fig3_others.log
for ds in arab rice rgb; do
  for part in 0-4 5-9; do
    folds=$(echo $part | tr '-' ',')
    setsid nohup $PY knode_cond_ladder_$ds.py --folds $folds --n_epochs 1000 \
      > $L/_fig3_${ds}_${part}.log 2>&1 < /dev/null &
    echo "  $ds folds=$folds PID=$!" >> $L/_fig3_others.log
  done
done

# ?????, ???????
while pgrep -f "knode_cond_ladder_" > /dev/null; do sleep 30; done
for ds in arab rice rgb; do
  echo "[$(date '+%T')] ?? $ds" >> $L/_fig3_others.log
  $PY knode_cond_ladder_$ds.py --aggregate >> $L/_fig3_others.log 2>&1
done
echo "[$(date '+%T')] DONE_FIG3_OTHERS" >> $L/_fig3_others.log
