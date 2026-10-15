#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
echo "=== ??? (cond,rank) ??? (??? 5 cond x 11 rank = 55) ==="
for i in 0 1 2 3; do
  n=$(grep -c "] h2_U=" $L/_cpu_work$i.log 2>/dev/null)
  fold=$(grep -o "===== fold [0-9]*" $L/_cpu_work$i.log | tail -1)
  echo "  cpu_work$i: $n / 55   ($fold)"
done
a=$(grep -h "] h2_U=" $L/_cpu_work*.log 2>/dev/null | wc -l)
sleep 180
b=$(grep -h "] h2_U=" $L/_cpu_work*.log 2>/dev/null | wc -l)
echo
echo "180s ??? $((b-a)) ??? -> ?? $(awk "BEGIN{printf \"%.2f\", ($b-$a)/180}") ?/?"
echo "??(55?)???? $(awk "BEGIN{if ($b-$a>0) printf \"%.1f\", 55/(($b-$a)/180)/60; else print \"NA\"}") ??"
echo "?? 4 ? fold ?????? $(awk "BEGIN{if ($b-$a>0) printf \"%.1f\", (220-$b)/(($b-$a)/180)/3600; else print \"NA\"}") ?? (220 = 4? x 55)"
date '+?? %H:%M:%S'
