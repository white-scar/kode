#!/bin/bash
KD=/root/autodl-tmp/qkn/outputs/kode_result/arab
echo "=== 1) ???? arab ???? ==="
ls $KD/ | grep "all_summary_hp_" | sed 's/knode_arab_tp1skip_all_summary_//;s/.csv//' | tr '\n' ' '
echo; echo "   ? $(ls $KD/ | grep -c 'all_summary_hp_') ?"
echo
echo "=== 2) ?? summary ?? mode ?????????? _skip?==="
for f in $KD/knode_arab_tp1skip_all_summary_hp_base.csv $KD/knode_arab_tp1skip_all_summary_hp_dim40.csv; do
  echo "-- $(basename $f)"
  cat $f
done
echo
echo "=== 3) base ??? per_trait ????? mode ==="
head -1 $KD/knode_arab_tp1skip_ode_hyper_hp_base_per_trait.csv | tr ',' '\n' | head -12
echo "   ????:"
ls $KD/ | grep "hp_base" 
