#!/bin/bash
# _dg_rank_all.sh -- arab / rice / rgb(drought) 三个数据集的 D_G x rank 全量扫描 (清空重跑版)
# 网格: DG=[10,20,30,40,50,80] x rank=[1,2,3,4] = 24 cell; 每数据集 24x10x5 = 1200 个模型
# 每折一个进程 (共 30); 全部结束后串行 aggregate; 最后写 _dg_rank_all.DONE
set -u
cd /root/autodl-tmp/qkn || exit 9
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOGD=outputs/reproduce/_dgrank_logs

# ---- 0) 杀掉上一轮残留, 保证没有旧进程在写文件 ----
pat='dg_rank''_joint'
if [ "$(pgrep -fc "$pat")" -gt 0 ]; then
  echo "[$(date '+%F %T')] killing leftover workers"
  pkill -f "$pat"
  sleep 5
fi
# ---- 1) 清空上一轮生成的所有结果 ----
rm -rf outputs/reproduce/dg_rank_arab outputs/reproduce/dg_rank_rice outputs/reproduce/dg_rank_rgb
rm -rf "$LOGD" outputs/reproduce/_dg_rank_all.DONE
mkdir -p "$LOGD"
: > _dg_rank_all.log
echo "[$(date '+%F %T')] cleaned old dg_rank outputs; START dg_rank all" >> _dg_rank_all.log

declare -A SCRIPTS=(
  [arab]=reproduce/knode_arab_dg_rank_joint.py
  [rice]=reproduce/knode_rice_dg_rank_joint.py
  [rgb]=reproduce/knode_rgb_dg_rank_joint.py
)

for ds in arab rice rgb; do
  for f in 0 1 2 3 4 5 6 7 8 9; do
    nohup $PY ${SCRIPTS[$ds]} --run --folds $f > "$LOGD/${ds}_f${f}.log" 2>&1 &
    echo "[$(date '+%F %T')] launched $ds fold $f pid=$!" >> _dg_rank_all.log
  done
done
echo "[$(date '+%F %T')] 30 processes launched" >> _dg_rank_all.log

wait
echo "[$(date '+%F %T')] all training done; aggregating" >> _dg_rank_all.log

for ds in arab rice rgb; do
  $PY ${SCRIPTS[$ds]} --aggregate-only > "$LOGD/${ds}_agg.log" 2>&1
  echo "[$(date '+%F %T')] aggregate $ds rc=$?" >> _dg_rank_all.log
done

date '+%F %T' > outputs/reproduce/_dg_rank_all.DONE
echo "[$(date '+%F %T')] DONE" >> _dg_rank_all.log
