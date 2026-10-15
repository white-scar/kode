#!/bin/bash
# _rerun_1000.sh -- (A) 图七 4 份 @1000 epoch 重跑 (旧 800ep 结果先归档到 _legacy_ep800_20260920/)
#                   (B) 图六 补齐缺失 config (arab 全 16 / uav 全 18 / rice 缺 13), pass1 并行 CONC=12 + pass2 顺序补缺
#                   (C) 图六聚合 -> outputs/reproduce/hyperparam_ablation_<ds>_summary.csv
# rc 一律用 `rc=$?` 单独一行取 (不能和 $(date) 写在同一行, 否则 $? 被命令替换吃掉)
set -u
cd /root/autodl-tmp/qkn/reproduce || exit 9
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
QDIR=/root/autodl-tmp/qkn
LOGD=$QDIR/outputs/reproduce/_rerun1000_logs; mkdir -p "$LOGD"
LOG=$QDIR/_rerun_1000.log
: > $LOG

GRID_ARAB="base dim10 dim20 dim40 dim50 hid32 hid64 hid256 depth2 depth3 act_gelu act_relu act_linear seed1234 seed2024 seed2025"
GRID_RICE="base dim5 dim20 dim30 hid32 hid256 depth2 depth3 act_gelu act_relu seed1234 seed2024 seed2025"
GRID_UAV="base dim10 dim20 dim30 dim40 dim80 dim100 hid32 hid64 hid256 depth2 depth3 act_gelu act_relu act_linear seed1234 seed2024 seed2025"
export GRID_ARAB GRID_RICE GRID_UAV
OD_ARAB=arab; OD_RICE=rice_indica_control; OD_UAV=maize_rgb_drought
PF_ARAB=knode_arab_tp1_; PF_RICE=knode_rice_tp1_; PF_UAV=knode_maize_rgb_tp1_

echo "[$(date '+%F %T')] ==== A) fig7 rerun @1000 ====" >> $LOG
for d in knode_extrap_arab_nt2 knode_extrap_rice_nt2 knode_extrap_rgb_nt2 knode_extrap_maize_nt5; do
  mkdir -p "../outputs/reproduce/$d/_legacy_ep800_20260920"
  mv ../outputs/reproduce/$d/*.csv "../outputs/reproduce/$d/_legacy_ep800_20260920/" 2>/dev/null
done
run_fig7() {
  local tag="$1" sc="$2"; shift 2
  echo "[$(date '+%F %T')] START fig7 $tag" >> $LOG
  $PY "$sc" "$@" > "$LOGD/fig7_$tag.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] END   fig7 $tag rc=$rc" >> $LOG
}
export -f run_fig7
( run_fig7 maize arab/knode_extrap_maize_arab_v2.py --dataset maize --n_test 5 --epochs 1000 ) &
( run_fig7 arab  arab/knode_extrap_maize_arab_v2.py --dataset arab  --n_test 2 --epochs 1000 ) &
( run_fig7 rice  rice/knode_extrap_maize_rice_v2.py  --dataset rice  --n_test 2 --epochs 1000 ) &
( run_fig7 uav   knode_extrap_maize_rgb_v2.py        --dataset rgb --env drought --n_test 2 --epochs 1000 ) &
wait
echo "[$(date '+%F %T')] ==== A) fig7 done ====" >> $LOG

echo "[$(date '+%F %T')] ==== B) fig6 hp fill-in ====" >> $LOG
mkdir -p ../outputs/kode_result/_junk_time50
mv ../outputs/kode_result/arab/*_time50* ../outputs/kode_result/maize_rgb_drought/*_time50* ../outputs/kode_result/_junk_time50/ 2>/dev/null
run_hp() {
  local ds="$1" cfg="$2"
  $PY knode_hyperparam_ablation.py --dataset "$ds" --_single "$cfg" \
      --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper \
      --tag "_hp_${cfg}" > "$LOGD/hp_${ds}_${cfg}.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] hp $ds $cfg rc=$rc" >> $LOG
}
export -f run_hp
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
date '+%F %T' > $QDIR/outputs/reproduce/_rerun_1000.DONE
echo "[$(date '+%F %T')] ALL DONE" >> $LOG