#!/bin/bash
# ?3 ???? ? ? skip ? 10 ? 5 ??? + ????
R=/root/autodl-tmp/qkn/reproduce
L=/root/autodl-tmp/qkn/outputs/reproduce
PY=/root/miniconda3/bin/python
cd $R || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
rm -f $L/knode_cond_ladder_maize_*.csv
: > $L/_fig3_condladder.log
echo "[$(date '+%T')] ?? 5 ??? (? 2 ?, 1000 epoch, ? skip)" >> $L/_fig3_condladder.log
for pair in "0,1" "2,3" "4,5" "6,7" "8,9"; do
  setsid nohup $PY knode_cond_ladder_maize.py --folds $pair --n_epochs 1000 \
    > $L/_fig3_f${pair//,/}.log 2>&1 < /dev/null &
  echo "  folds=$pair PID=$!" >> $L/_fig3_condladder.log
done

# ????????
while pgrep -f "knode_cond_ladder_maize.py" > /dev/null; do sleep 20; done
echo "[$(date '+%T')] ?????, ????" >> $L/_fig3_condladder.log
$PY knode_cond_ladder_maize.py --aggregate >> $L/_fig3_condladder.log 2>&1
echo "[$(date '+%T')] DONE_FIG3" >> $L/_fig3_condladder.log
