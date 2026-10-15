#!/bin/bash
cd /root/autodl-tmp/qkn/reproduce || exit 1
PY=/root/miniconda3/bin/python
for s in arab rice rgb; do
  echo "===== SMOKE $s ====="
  timeout 900 $PY knode_cond_ladder_$s.py --folds 0 --n_epochs 20 2>&1 | tail -8
done
echo SMOKE_ALL_DONE
