#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
a=$(grep -o "inner . done" $L/_hp_work*.log 2>/dev/null | wc -l)
sleep 120
b=$(grep -o "inner . done" $L/_hp_work*.log 2>/dev/null | wc -l)
echo "hp ??: $(awk "BEGIN{printf \"%.2f\", ($b-$a)/120}") ?/? (? CPU ???? ~0.32)"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"
date '+?? %H:%M:%S'
