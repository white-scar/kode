#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
echo "=== 4 ????? ==="
pgrep -af knode_extrap | sed -E 's#/root/miniconda3/bin/python ##' | sed -E 's/--foldsets [0-9,]+/--foldsets ../' | sed -E 's/--dataset /ds=/' | sed -E 's/ knode_extrap_maize_arab_v2.py//'
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo
for f in maize arab rgb_drought rgb_irrigated; do
  n=$(grep -c "foldset " $L/_extrap_${f}_noskip.log 2>/dev/null)
  echo "-- $f : ??? foldset ? $((n/3)) / 10"
done
echo
echo "=== ??????? ==="
for f in maize arab rgb_drought rgb_irrigated; do
  printf "   %-14s %s\n" "$f" "$(grep 'foldset ' $L/_extrap_${f}_noskip.log 2>/dev/null | tail -1 | cut -c1-95)"
done
date '+?? %H:%M:%S'
