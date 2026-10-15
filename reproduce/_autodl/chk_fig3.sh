#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
date '+?? %T'
echo "=== ?????3?? ==="
pgrep -f knode_cond_ladder_maize.py | wc -l
echo "=== ????? (?? 3 ???) ==="
for i in 0 1 2 3 4 5 6 7 8 9; do
  f=$L/knode_cond_ladder_maize_f$i.csv
  [ -f "$f" ] && printf "   fold%s ?  %s\n" $i "$(date -r $f '+%H:%M:%S')"
done
echo "=== ??? ==="
cat $L/_fig3_condladder.log
echo "=== ? worker ?? ==="
for p in 01 23 45 67 89; do
  printf "   f%s: %s\n" "$p" "$(tail -1 $L/_fig3_f$p.log 2>/dev/null | cut -c1-90)"
done
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
