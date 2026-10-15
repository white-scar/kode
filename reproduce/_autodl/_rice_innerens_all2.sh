#!/bin/bash
# _rice_innerens_all2.sh -- author_all2_innerens_rice.R 全 10 折并行 (修好聚合 bug 后重跑)
set -u
cd /root/autodl-tmp/qkn || exit 9
LOG=/root/autodl-tmp/qkn/_author_rice_innerens_all.log
mkdir -p outputs/reproduce/_cv3_logs
: > $LOG
echo "[$(date '+%F %T')] START rice innerens folds 1-10 (fixed aggregation)" >> $LOG
run_fold() {
  local k="$1"
  Rscript reproduce/author_all2_innerens_rice.R "$k" "$k" 2 6 outputs/reproduce/author_all2_rice_innerens > outputs/reproduce/_cv3_logs/rice_innerens_k${k}.log 2>&1
  echo "[$(date '+%F %T')] innerens fold $k rc=$?" >> $LOG
}
export -f run_fold; export LOG
for k in 1 2 3 4 5 6 7 8 9 10; do ( run_fold $k ) & done
wait
echo "[$(date '+%F %T')] folds 1-10 done" >> $LOG
date '+%F %T' > /root/autodl-tmp/qkn/outputs/reproduce/_author_rice_innerens.DONE