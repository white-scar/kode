#!/bin/bash
sleep 60
L=/root/autodl-tmp/qkn/outputs/reproduce
echo "=== 4 ?????????? 10 ??==="
for f in maize arab rgb_drought rgb_irrigated; do
  n=$(grep -c "foldset " $L/_extrap_${f}_noskip.log 2>/dev/null)
  last=$(grep "foldset " $L/_extrap_${f}_noskip.log 2>/dev/null | tail -1 | cut -c1-110)
  printf "   %-14s %2d/10 ?   %s\n" "$f" $((n/3)) "$last"
done
echo
echo "=== ??? / GPU ==="
echo "   ????: $(pgrep -f knode_extrap | wc -l)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
free -g | sed -n '2p' | awk '{print "   ?? used/avail: "$3"G / "$7"G"}'
date '+?? %H:%M:%S'
