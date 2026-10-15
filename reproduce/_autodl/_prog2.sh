#!/bin/bash
cd /root/autodl-tmp/qkn
date
echo "--- arab: 每折已完成的 MegaLMM 拟合数 (每折需 15 个) ---"
for k in $(seq 1 10); do
  d=outputs/reproduce/dynamicGP_MegaLMM_arabidopsis_CV3_unified/fold$k
  n=$(ls -d $d/run_* 2>/dev/null | wc -l)
  echo "  arab fold$k : $n"
done
echo "--- rice ---"
for k in $(seq 1 10); do
  d=outputs/reproduce/dynamicGP_MegaLMM_rice_CV3_unified/fold$k
  n=$(ls -d $d/run_* 2>/dev/null | wc -l)
  echo "  rice fold$k : $n"
done
echo "--- rice fig2 stage2 (每折 5 inner x 6 拟合 = 30) ---"
d=outputs/reproduce/author_all2_rice_innerens
ls -d $d/run_* 2>/dev/null | wc -l
grep -a -c "inner" _author_rice_innerens_smoke.log
echo "--- load ---"
cat /proc/loadavg