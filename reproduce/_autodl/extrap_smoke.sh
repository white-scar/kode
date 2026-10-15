#!/bin/bash
cd /root/autodl-tmp/qkn/reproduce || exit 1
PY=/root/miniconda3/bin/python
for DS in maize arab; do
  echo "===== SMOKE $DS (1 fold, 20 epochs, no-skip) ====="
  timeout 900 $PY knode_extrap_maize_arab_v2.py --dataset $DS --n_test 5 --foldsets 1 --epochs 20 --tag _smoke 2>&1 | tail -8
done
echo "--- ?? ---"
ls -la /root/autodl-tmp/qkn/outputs/reproduce/knode_extrap_maize_arab/ | grep smoke
echo SMOKE_DONE
