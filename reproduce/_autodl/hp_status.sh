#!/bin/bash
KD=/root/autodl-tmp/qkn/outputs/kode_result
LOGD=/root/autodl-tmp/qkn/outputs/reproduce
echo "=== ???? hp ?? ==="
pgrep -c -f knode_hyperparam_ablation.py
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== ?????? / ?? ==="
tot=0
for D in arab rice_indica_control maize_rgb_drought; do
  n=$(ls $KD/$D/ 2>/dev/null | grep -c "all_summary_hp_")
  echo "   $D : $n"
  tot=$((tot+n))
done
echo "   ?? $tot / 50"
echo "=== ?? worker ???? ==="
for i in 0 1 2 3 4 5 6 7; do
  printf "  work%s: %s\n" "$i" "$(tail -1 $LOGD/_hp_work$i.log 2>/dev/null)"
done
echo "=== ??????? ==="
ls -lt --time-style='+%H:%M' $KD/arab/*all_summary_hp_*.csv $KD/rice_indica_control/*all_summary_hp_*.csv $KD/maize_rgb_drought/*all_summary_hp_*.csv 2>/dev/null | head -6 | awk '{print "   "$6" "$7}'
