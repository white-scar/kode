#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
a=$(grep -o "inner . done" $L/_hp_work*.log 2>/dev/null | wc -l)
sleep 150
b=$(grep -o "inner . done" $L/_hp_work*.log 2>/dev/null | wc -l)
r=$((b-a))
date '+??: %H:%M:%S'
echo "150s ??? $r ? -> ?? $(awk "BEGIN{printf \"%.2f\", $r/150}") ?/?"
echo "?? $((5000-b)) ? -> ???? $(awk "BEGIN{printf \"%.2f\", (5000-$b)/($r/150)/3600}") ??"
