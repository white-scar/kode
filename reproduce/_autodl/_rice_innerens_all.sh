#!/bin/bash
cd /root/autodl-tmp/qkn
date
echo "--- stage2 fold1 产物 ---"
ls outputs/reproduce/author_all2_rice_innerens/components/ | head -4
cat outputs/reproduce/author_all2_rice_innerens/author_all2_innerens.csv 2>/dev/null | head -6
echo "--- 启动 folds 2-10 并行 ---"
LOG=/root/autodl-tmp/qkn/_author_rice_innerens_all.log
: > $LOG
echo "[$(date '+%F %T')] START rice innerens folds 2-10" >> $LOG
run_fold() {
  local k="$1"
  Rscript reproduce/author_all2_innerens_rice.R "$k" "$k" 2 6 outputs/reproduce/author_all2_rice_innerens > outputs/reproduce/_cv3_logs/rice_innerens_k${k}.log 2>&1
  echo "[$(date '+%F %T')] innerens fold $k rc=$?" >> $LOG
}
export -f run_fold; export LOG
for k in 2 3 4 5 6 7 8 9 10; do ( run_fold $k ) & done
wait
echo "[$(date '+%F %T')] folds 2-10 done" >> $LOG
date '+%F %T' > /root/autodl-tmp/qkn/outputs/reproduce/_author_rice_innerens.DONE