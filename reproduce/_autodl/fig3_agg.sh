#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
for ds in arab rice rgb; do
  echo "===== ?3 ? $ds ?10 ?, ? skip?====="
  cat $L/knode_cond_ladder_${ds}_agg.csv 2>/dev/null
  echo
done
echo "=== ???????????==="
cat $L/knode_cond_ladder_maize_agg.csv 2>/dev/null
