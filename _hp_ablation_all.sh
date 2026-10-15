#!/bin/bash
# _hp_ablation_all.sh -- 图六 超参数消融 (arab / rice / uav-drought)
#   每个 (数据集, config) 一个进程, 每进程 10 foldset x 2 mode x 5 inner = 100 个模型
#   输出: outputs/kode_result/<ds>/knode_*_tp1_all_summary_hp_<cfg>.csv / _per_trait / _all_detail
#   用 --_single 逐个跑 (不走 aggregate, 避免并发互相覆盖汇总表)
#   先等 rank sweep 结束 (显存), 跑完写 _hp_ablation.DONE
set -u
cd /root/autodl-tmp/qkn/reproduce || exit 9
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOGD=/root/autodl-tmp/qkn/outputs/reproduce/_hp_logs; mkdir -p "$LOGD"
CONC=30
LOG=/root/autodl-tmp/qkn/_hp_ablation_all.log
: > $LOG
echo "[$(date '+%F %T')] waiting for rank sweep to finish ..." >> $LOG
for i in $(seq 1 720); do [ -f /root/autodl-tmp/qkn/outputs/reproduce/_rank_sweep.DONE ] && break; sleep 30; done

GRID_ARAB="base dim10 dim20 dim40 dim50 hid32 hid64 hid256 depth2 depth3 act_gelu act_relu act_linear seed1234 seed2024 seed2025"
GRID_RICE="base dim5 dim20 dim30 dim50 hid32 hid64 hid256 depth2 depth3 act_gelu act_relu act_linear seed1234 seed2024 seed2025"
GRID_UAV="base dim10 dim20 dim30 dim40 dim80 dim100 hid32 hid64 hid256 depth2 depth3 act_gelu act_relu act_linear seed1234 seed2024 seed2025"

run_cfg() {   # $1=ds $2=cfg
  local ds="$1" cfg="$2"
  $PY knode_hyperparam_ablation.py --dataset "$ds" --_single "$cfg" \
      --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper \
      --tag "_hp_${cfg}" > "$LOGD/${ds}_${cfg}.log" 2>&1
  echo "[$(date '+%F %T')] done $ds $cfg rc=$?" >> $LOG
}
export -f run_cfg; export PY LOGD LOG

: > /root/autodl-tmp/qkn/_hp_jobs.txt
for c in $GRID_ARAB; do echo "arab $c" >> /root/autodl-tmp/qkn/_hp_jobs.txt; done
for c in $GRID_RICE; do echo "rice $c" >> /root/autodl-tmp/qkn/_hp_jobs.txt; done
for c in $GRID_UAV;  do echo "uav  $c" >> /root/autodl-tmp/qkn/_hp_jobs.txt; done
echo "[$(date '+%F %T')] jobs = $(wc -l < /root/autodl-tmp/qkn/_hp_jobs.txt)" >> $LOG
xargs -P $CONC -n 2 -a /root/autodl-tmp/qkn/_hp_jobs.txt bash -c 'run_cfg "$@"' _
date '+%F %T' > /root/autodl-tmp/qkn/outputs/reproduce/_hp_ablation.DONE
echo "[$(date '+%F %T')] DONE" >> $LOG
