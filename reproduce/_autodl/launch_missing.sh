#!/bin/bash
R=/root/autodl-tmp/qkn/reproduce
L=/root/autodl-tmp/qkn/outputs/reproduce
PY=/root/miniconda3/bin/python
cd $R || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# ????/????
pkill -f "tag _time" 2>/dev/null
rm -f $L/knode_missing_maize_arab/*_smoke* $L/knode_missing_maize_arab/*_time*

# ?9 ??? (? skip): ?? + ???, ?? scenario, 10 ?, 800 epoch
setsid nohup $PY knode_missing_maize_arab_v2.py --dataset maize \
  --foldsets 1,2,3,4,5,6,7,8,9,10 --epochs 800 > $L/_missing_maize_noskip.log 2>&1 < /dev/null &
echo "missing-maize PID=$!"
setsid nohup $PY knode_missing_maize_arab_v2.py --dataset arab \
  --foldsets 1,2,3,4,5,6,7,8,9,10 --epochs 800 > $L/_missing_arab_noskip.log 2>&1 < /dev/null &
echo "missing-arab  PID=$!"

sleep 8
echo "=== ????????? ==="
pgrep -af "knode_extrap|knode_missing" | sed -E 's#/root/miniconda3/bin/python ##' | sed -E 's/(--foldsets|--scen) [0-9,a-z]+/\1 .../' | sed -E 's/--epochs [0-9]+/--epochs 800/'
echo "???: $(pgrep -f 'knode_extrap|knode_missing' | wc -l)"
echo "=== ????? ==="
pgrep -af autoshutdown2.sh
date '+?? %T'
