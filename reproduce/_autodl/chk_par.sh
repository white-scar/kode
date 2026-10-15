#!/bin/bash
sleep 90
L=/root/autodl-tmp/qkn/outputs/reproduce
echo "=== ???????? ==="
pgrep -f knode_extrap | wc -l
pgrep -af knode_extrap | sed -E 's/--foldsets [0-9,]+/--foldsets .../' | sed -E 's#/root/miniconda3/bin/python ##'
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
echo
for f in _extrap_maize_noskip _extrap_arab_noskip _extrap_rgb_drought_noskip _extrap_rgb_irrigated_noskip; do
  echo "-- $f"; tail -2 $L/$f.log 2>/dev/null | sed 's/^/     /'
done
date '+%H:%M:%S'
