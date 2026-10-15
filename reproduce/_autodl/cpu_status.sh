#!/bin/bash
sleep 120
L=/root/autodl-tmp/qkn/outputs/reproduce
echo "=== CPU worker ??? ==="
for i in 0 1 2 3; do echo "-- cpu_work$i"; tail -3 $L/_cpu_work$i.log; done
echo "=== ???? ==="
grep -l -iE "Traceback|Error" $L/_cpu_work*.log 2>/dev/null | head -5
echo "=== ??/?? ==="
echo "knode ???: $(pgrep -c -f 'knode_')"
echo "loadavg: $(cat /proc/loadavg)"
echo "CPU idle: $(top -bn1 | sed -n '3p' | awk '{print $8}')"
echo "????: $(free -g | awk 'NR==2{print $7}') GB"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
