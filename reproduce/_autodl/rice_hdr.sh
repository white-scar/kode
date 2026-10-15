#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
echo "=== rice rank per-fold ???? ==="
head -2 $L/knode_rank_genetics_value_rice_f0.csv
echo
echo "=== ?? ==="
wc -l $L/knode_rank_genetics_value_rice_f0.csv
