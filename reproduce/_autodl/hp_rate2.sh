#!/bin/bash
sleep 90
L=/root/autodl-tmp/qkn/outputs/reproduce
a=$(grep -o "inner . done" $L/_hp_work*.log 2>/dev/null | wc -l)
sleep 150
b=$(grep -o "inner . done" $L/_hp_work*.log 2>/dev/null | wc -l)
echo "hp ????: $(awk "BEGIN{printf \"%.2f\", ($b-$a)/150}") ?/?  (8?????? 0.32)"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"
echo "loadavg: $(cat /proc/loadavg)"
echo "?????: $(grep -h 'END   ' $L/_hp_work*.log 2>/dev/null | wc -l) / 50"
date '+?? %H:%M:%S'
