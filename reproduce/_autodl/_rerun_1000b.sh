#!/bin/bash
# _rerun_1000b.sh -- 只做图六补齐 (B) + 聚合 (C); 图七已由 _rerun_1000.sh 的 A) 段跑完 @1000, 不再重跑
# 修正 _rerun_1000.sh 的 bug: run_hp 用到的 PY/LOGD/LOG 没 export, xargs 子 shell 里为空 -> pass1 47 个 job 秒挂
set -u
cd /root/autodl-tmp/qkn/reproduce || exit 9
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
QDIR=/root/autodl-tmp/qkn
LOGD=$QDIR/outputs/reproduce/_rerun1000_logs; mkdir -p "$LOGD"
LOG=$QDIR/_rerun_1000b.log
: > $LOG

GRID_ARAB="base dim10 dim20 dim40 dim50 hid32 hid64 hid256 depth2 depth3 act_gelu act_relu act_linear seed1234 seed2024 seed2025"
GRID_RICE="base dim5 dim20 dim30 hid32 hid256 depth2 depth3 act_gelu act_relu seed1234 seed2024 seed2025"
GRID_UAV="base dim10 dim20 dim30 dim40 dim80 dim100 hid32 hid64 hid256 depth2 depth3 act_gelu act_relu act_linear seed1234 seed2024 seed2025"
OD_ARAB=arab; OD_RICE=rice_indica_control; OD_UAV=maize_rgb_drought
PF_ARAB=knode_arab_tp1_; PF_RICE=knode_rice_tp1_; PF_UAV=knode_maize_rgb_tp1_
export PY LOGD LOG QDIR GRID_ARAB GRID_RICE GRID_UAV OD_ARAB OD_RICE OD_UAV PF_ARAB PF_RICE PF_UAV

run_hp() {
  local ds="$1" cfg="$2"
  $PY knode_hyperparam_ablation.py --dataset "$ds" --_single "$cfg" \
      --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper \
      --tag "_hp_${cfg}" > "$LOGD/hp_${ds}_${cfg}.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] hp $ds $cfg rc=$rc" >> $LOG
}
export -f run_hp

echo "[$(date '+%F %T')] ==== B) fig6 hp fill-in (fix: env exported) ====" >> $LOG
: > $QDIR/_hp_jobs2.txt
for c in $GRID_ARAB; do echo "arab $c" >> $QDIR/_hp_jobs2.txt; done
for c in $GRID_RICE; do echo "rice $c" >> $QDIR/_hp_jobs2.txt; done
for c in $GRID_UAV;  do echo "uav  $c" >> $QDIR/_hp_jobs2.txt; done
echo "[$(date '+%F %T')] pass1 jobs=$(wc -l < $QDIR/_hp_jobs2.txt) CONC=12" >> $LOG
xargs -P 12 -n 2 -a $QDIR/_hp_jobs2.txt bash -c 'run_hp "$@"' _
echo "[$(date '+%F %T')] pass1 done" >> $LOG

is_missing() {
  local od pf
  case "$1" in
    arab) od=$OD_ARAB; pf=$PF_ARAB;;
    rice) od=$OD_RICE; pf=$PF_RICE;;
    uav)  od=$OD_UAV;  pf=$PF_UAV;;
  esac
  [ -f "../outputs/kode_result/$od/${pf}all_summary_hp_$2.csv" ] && return 1
  return 0
}
while read -r ds cfg; do
  if is_missing "$ds" "$cfg"; then
    echo "[$(date '+%F %T')] pass2 redo $ds $cfg" >> $LOG
    run_hp "$ds" "$cfg"
  fi
done < $QDIR/_hp_jobs2.txt
echo "[$(date '+%F %T')] pass2 done" >> $LOG

$PY _hp_agg.py arab "$GRID_ARAB" >> $LOG 2>&1
$PY _hp_agg.py rice "$GRID_RICE" >> $LOG 2>&1
$PY _hp_agg.py uav  "$GRID_UAV"  >> $LOG 2>&1
date '+%F %T' > $QDIR/outputs/reproduce/_rerun_1000b.DONE
echo "[$(date '+%F %T')] ALL DONE" >> $LOG