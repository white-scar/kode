#!/bin/bash
echo "===== AutoDL 4090 ====="
date '+  ?? %m-%d %H:%M:%S'
L=/root/autodl-tmp/qkn/outputs/reproduce
KD=/root/autodl-tmp/qkn/outputs/kode_result
echo "  hp ??: $(pgrep -c -f knode_hyperparam_ablation)"
echo "  rank ??: $(pgrep -c -f knode_rank)"
echo "  GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
echo "  loadavg: $(cat /proc/loadavg)"
echo "  --- hp ????? ---"
for D in arab rice_indica_control maize_rgb_drought; do
  n=$(ls $KD/$D/ 2>/dev/null | grep -c "all_summary_hp_")
  echo "     $D : $n"
done
echo "     hp ??: $(ls $KD/*/ 2>/dev/null | grep -c 'all_summary_hp_') / 50"
echo "  --- CPU rank ????? 55 ????---"
for i in 0 1 2 3; do
  n=$(grep -c "] h2_U=" $L/_cpu_work$i.log 2>/dev/null)
  fold=$(grep -o "===== fold [0-9]*" $L/_cpu_work$i.log 2>/dev/null | tail -1)
  echo "     cpu_work$i: $n/55  ($fold)"
done
echo "  --- ???? rank/?? fold ? ---"
ls $L/knode_rank_genetics_value_rice_f*.csv $L/knode_rank_genetics_value_uav_f*.csv 2>/dev/null | wc -l | sed 's/^/     rank fold: /'
ls $L/rank_select_v2_skipaffine_*.csv 2>/dev/null | wc -l | sed 's/^/     ?? fold: /'
