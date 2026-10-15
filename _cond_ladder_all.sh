#!/bin/bash
# _cond_ladder_all.sh -- Fig.3 cond ladder 十折 (arab / rice / rgb-drought)
#   3 数据集 x 10 fold = 30 进程; 每进程 4 个需训练档 x 5 inner = 20 个模型 (persistence 不训练)
#   dim/rank 用锁定值: arab D_G=30 r=2 / rice D_G=10 r=2 / rgb D_G=50 r=2
#   输出: outputs/reproduce/cond_ladder_<ds>_dg<DG>_r<rank>/{f0..9.csv, trait_f*.csv, agg.csv, ...}
set -u
cd /root/autodl-tmp/qkn || exit 9
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOGD=outputs/reproduce/_condladder_logs
mkdir -p "$LOGD"
: > _cond_ladder_all.log
echo "[$(date '+%F %T')] START cond ladder (arab/rice/rgb x 10 fold)" >> _cond_ladder_all.log

launch() {   # $1=tag  $2=script
  for f in 0 1 2 3 4 5 6 7 8 9; do
    nohup $PY "$2" --folds $f > "$LOGD/$1_f$f.log" 2>&1 &
    echo "[$(date '+%F %T')] launched $1 fold $f pid=$!" >> _cond_ladder_all.log
  done
}

launch arab reproduce/arab/knode_cond_ladder_arab.py
launch rice reproduce/rice/knode_cond_ladder_rice.py
launch rgb  reproduce/maize_RGB/knode_cond_ladder_rgb.py
echo "[$(date '+%F %T')] 30 processes launched" >> _cond_ladder_all.log

wait
for kv in "arab reproduce/arab/knode_cond_ladder_arab.py" \
          "rice reproduce/rice/knode_cond_ladder_rice.py" \
          "rgb reproduce/maize_RGB/knode_cond_ladder_rgb.py"; do
  set -- $kv
  $PY "$2" --aggregate > "$LOGD/$1_agg.log" 2>&1
  echo "[$(date '+%F %T')] aggregate $1 rc=$?" >> _cond_ladder_all.log
done
date '+%F %T' > outputs/reproduce/_cond_ladder_all.DONE
echo "[$(date '+%F %T')] DONE" >> _cond_ladder_all.log
