#!/bin/bash
echo "CPU ??: $(nproc)"
echo "loadavg : $(cat /proc/loadavg)"
top -bn1 | sed -n '3p'
free -g | head -2
echo "--- ? KNode ???? ---"
pgrep -c -f 'knode_'
echo "--- ? 3 ??????? ---"
for p in $(pgrep -f knode_hyperparam_ablation.py | head -3); do
  echo "  pid $p  threads=$(ls /proc/$p/task 2>/dev/null | wc -l)"
done
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
