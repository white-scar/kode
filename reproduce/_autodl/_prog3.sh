#!/bin/bash
cd /root/autodl-tmp/qkn
date
echo "--- innerens 每折已完成拟合数 (每折 30) ---"
for k in 1 2 3 4 5 6 7 8 9 10; do
  n=$(ls -d outputs/reproduce/author_all2_rice_innerens/run_$(printf %02d $k)_inner* 2>/dev/null | wc -l)
  echo "  fold$k : $n"
done
echo "--- 是否有报错 ---"
grep -a -l "Error" outputs/reproduce/_cv3_logs/rice_innerens_k*.log 2>/dev/null | head
echo "--- 已完成折产物 ---"
ls outputs/reproduce/author_all2_rice_innerens/components/ 2>/dev/null | wc -l
cat _author_rice_innerens_all.log
echo "--- 分步耗时样本 (fold1 最新一条) ---"
grep -a "Time difference" outputs/reproduce/_cv3_logs/rice_innerens_k1.log | tail -3
echo "--- procs/load ---"
pgrep -fc "R --no-echo"; cat /proc/loadavg