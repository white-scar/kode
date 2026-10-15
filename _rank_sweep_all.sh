#!/bin/bash
# _rank_sweep_all.sh -- Fig.4 (rank sweep) 三个数据集: arab / rice / uav(drought)
#   每个 (数据集, fold, cond) 一个进程 => 3 x 10 x 4 = 120 个任务, 全局并发上限 CONC
#   每个任务内部把该 (fold, cond) 的所有 rank 跑完; cell csv 已存在会跳过 => 绝不覆盖已有结果
#   noise 档不跑 (图里不用); 以后要补可以单独补, 不影响已有 cell
set -u
cd /root/autodl-tmp/qkn || exit 9
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOGD=outputs/reproduce/_rank_sweep_logs; mkdir -p "$LOGD"
CONC=40
CONDS="full tp1only grmonly const"
: > _rank_sweep_all.log
echo "[$(date '+%F %T')] waiting for maize dg_rank to finish (VRAM) ..." >> _rank_sweep_all.log
for i in $(seq 1 360); do pgrep -f '[k]node_maize_dg_rank_joint' >/dev/null || break; sleep 20; done
echo "[$(date '+%F %T')] start rank sweep (CONC=$CONC)" >> _rank_sweep_all.log

run_job() {   # $1=ds $2=script $3=fold $4=cond $5=env(- 表示无)
  local ds="$1" sc="$2" f="$3" c="$4" env="$5"
  local extra=""; [ "$env" != "-" ] && extra="--env $env"
  $PY "$sc" --folds "$f" --conds "$c" $extra > "$LOGD/${ds}_f${f}_${c}.log" 2>&1
  echo "[$(date '+%F %T')] done $ds f$f $c rc=$?" >> _rank_sweep_all.log
}
export -f run_job; export PY LOGD

: > _jobs.txt
for f in 0 1 2 3 4 5 6 7 8 9; do
  for c in $CONDS; do
    echo "arab reproduce/arab/knode_rank_genetics_value_arab.py $f $c -" >> _jobs.txt
    echo "rice reproduce/rice/knode_rank_genetics_value_rice.py $f $c -" >> _jobs.txt
    echo "uav  reproduce/maize_RGB/knode_rank_genetics_value_uav.py $f $c drought" >> _jobs.txt
  done
done
echo "[$(date '+%F %T')] jobs = $(wc -l < _jobs.txt)" >> _rank_sweep_all.log
xargs -P $CONC -n 5 -a _jobs.txt bash -c 'run_job "$@"' _

$PY reproduce/arab/knode_rank_genetics_value_arab.py --aggregate > "$LOGD/agg_arab.log" 2>&1
$PY reproduce/rice/knode_rank_genetics_value_rice.py --aggregate > "$LOGD/agg_rice.log" 2>&1
$PY reproduce/maize_RGB/knode_rank_genetics_value_uav.py --aggregate --env drought > "$LOGD/agg_uav.log" 2>&1
date '+%F %T' > outputs/reproduce/_rank_sweep.DONE
echo "[$(date '+%F %T')] DONE" >> _rank_sweep_all.log
