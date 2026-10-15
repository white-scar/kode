#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
a=$(grep -o "inner . done" $L/_hp_work*.log 2>/dev/null | wc -l)
f1=$(grep -o "Fold-set [0-9]*/10" $L/_hp_work0.log | tail -1)
sleep 180
b=$(grep -o "inner . done" $L/_hp_work*.log 2>/dev/null | wc -l)
f2=$(grep -o "Fold-set [0-9]*/10" $L/_hp_work0.log | tail -1)
echo "180s ??? inner ???: $((b-a))  (8 ???)"
echo "=> ?? 1000-epoch ??? $(python3 -c "print(round(180/max($b-$a,1),1))") s"
echo "work0 fold-set: $f1 -> $f2"
