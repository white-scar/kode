#!/bin/bash
date '+???????: %H:%M:%S'
L=/root/autodl-tmp/qkn/outputs/reproduce
a=$(grep -o "inner . done" $L/_hp_work*.log 2>/dev/null | wc -l)
done=$(grep -c "END   " $L/_hp_work*.log 2>/dev/null | awk -F: '{s+=$2} END{print s}')
echo "??? inner ????: $a / 5000"
echo "??????: $done / 50"
echo "--- ? worker ?? ---"
for i in 0 1 2 3 4 5 6 7; do
  n=$(grep -c "END   " $L/_hp_work$i.log)
  cur=$(grep "START" $L/_hp_work$i.log | tail -1 | sed 's/.*START //')
  fs=$(grep -o "Fold-set [0-9]*/10" $L/_hp_work$i.log | tail -1)
  echo "  work$i: ?? $n ??? | ?? $cur | $fs"
done
