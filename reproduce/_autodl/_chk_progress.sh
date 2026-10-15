#!/bin/bash
cd /root/autodl-tmp/qkn
date
echo "--- arab CV3 log ---"
cat _cv3_arab.log
echo "--- rice CV3 log ---"
cat _cv3_rice.log
echo "--- completed folds ---"
ls outputs/reproduce/dynamicGP_MegaLMM_arabidopsis_CV3_unified/fold*/snapshot_metrics_by_time.csv 2>/dev/null | wc -l
ls outputs/reproduce/dynamicGP_MegaLMM_rice_CV3_unified/fold*/snapshot_metrics_by_time.csv 2>/dev/null | wc -l
echo "--- per-fold progress (arab k1, rice k1) ---"
tail -3 outputs/reproduce/_cv3_logs/arab_k1.log | tr -d '\r' | cut -c1-90
tail -3 outputs/reproduce/_cv3_logs/rice_k1.log | tr -d '\r' | cut -c1-90
echo "--- rice fig2 stage2 progress ---"
grep -a -E "inner [0-9]:|outer [0-9] done" _author_rice_innerens_smoke.log | tail -6
echo "--- procs / load ---"
pgrep -fc "R --no-echo"
cat /proc/loadavg
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader