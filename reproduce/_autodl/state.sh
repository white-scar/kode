#!/bin/bash
echo "=== ?? knode ?? python ?? ==="
ps -eo pid,ppid,etime,pcpu,cmd | grep "[k]node_" | sed 's/\(.\{150\}\).*/\1/'
echo
echo "=== ???? ==="
echo "  hp   : $(pgrep -c -f knode_hyperparam)"
echo "  rank : $(pgrep -c -f knode_rank)"
echo "  ?? : $(pgrep -c -f 'knode_' )"
echo
echo "=== CPU ?? ==="
echo "loadavg: $(cat /proc/loadavg)"
top -bn1 | sed -n '3p'
echo "??????: $(ps -eLf | awk '$8=="R"' | wc -l)"
echo
echo "=== ?? ==="
free -g | head -2
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo
echo "=== ????? / ?? ==="
L=/root/autodl-tmp/qkn/outputs/reproduce
echo "  hp ??: $(grep -h 'END   ' $L/_hp_work*.log 2>/dev/null | wc -l) / 50"
echo "  rank/????: $(ls $L/knode_rank_genetics_value_*.csv $L/rank_select_v2_skipaffine_*.csv 2>/dev/null | wc -l)"
echo "  CPU worker ??: $(pgrep -c -f 'cpu_work')"
date '+?? %H:%M:%S'
