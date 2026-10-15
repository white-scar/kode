#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
date '+?? %T'
echo "=== ????? ==="; pgrep -c -f knode_cond_ladder
echo "=== ???????? ==="
for f in $L/_fig3_arab_A.log $L/_fig3_arab_B.log $L/_fig3_rice_A.log $L/_fig3_rice_B.log $L/_fig3_rgb_A.log $L/_fig3_rgb_B.log; do
  n=$(basename $f); n=${n#_fig3_}; n=${n%.log}
  printf "   %-8s ???=%s | ??: %s\n" "$n" \
    "$(grep -c '===== fold' $f 2>/dev/null)" \
    "$(grep -E '^\s*\[full\]' $f 2>/dev/null | tail -1 | cut -c1-60)"
done
echo "=== ??????? ==="
for ds in arab rice rgb; do
  echo "   $ds: $(ls $L/knode_cond_ladder_${ds}_f*.csv 2>/dev/null | grep -v trait | wc -l)/10 ?"
done
echo "=== ???? ==="; tail -3 $L/_fig3_others.log
echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
