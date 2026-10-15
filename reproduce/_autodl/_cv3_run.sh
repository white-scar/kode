#!/bin/bash
# _cv3_run.sh -- 跑作者的 dynamicGP-MegaLMM+TP1 CV3 复现 (arab / rice)
#   用法: bash _cv3_run.sh arab [k0] [k1] [ncores]
#   每折一个独立进程 + 独立 OUTDIR (fold{k}), 跑完由 _cv3_agg.py 汇总
set -u
DS=${1:?arab|rice}; K0=${2:-1}; K1=${3:-10}; NC=${4:-8}
Q=/root/autodl-tmp/qkn
cd $Q || exit 9
TAG=$( [ "$DS" = "arab" ] && echo arabidopsis || echo rice )
if [ "$DS" = "arab" ]; then SC=reproduce/arab/dynamicGP_MegaLMM_TP1_arabidopsis_CV3.R; else SC=reproduce/rice/dynamicGP_MegaLMM_TP1_rice_CV3.R; fi
BASE=$Q/outputs/reproduce/dynamicGP_MegaLMM_${TAG}_CV3_unified
LOGD=$Q/outputs/reproduce/_cv3_logs; mkdir -p "$LOGD" "$BASE"
LOG=$Q/_cv3_${DS}.log
: > $LOG
echo "[$(date '+%F %T')] START $DS folds $K0-$K1 ncores=$NC" >> $LOG
run_fold() {
  local k="$1"
  Rscript "$SC" "$k" "$k" 2 "$NC" "$BASE/fold$k" > "$LOGD/${DS}_k${k}.log" 2>&1
  echo "[$(date '+%F %T')] fold $k rc=$?" >> $LOG
}
export -f run_fold; export SC BASE LOGD DS NC
for k in $(seq $K0 $K1); do ( run_fold $k ) & done
wait
echo "[$(date '+%F %T')] folds done, aggregate" >> $LOG
/root/miniconda3/bin/python $Q/reproduce/_cv3_agg.py "$DS" "$K0" "$K1" >> $LOG 2>&1
echo "[$(date '+%F %T')] DONE $DS" >> $LOG
date '+%F %T' > $Q/outputs/reproduce/_cv3_${DS}.DONE