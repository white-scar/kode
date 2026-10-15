#!/bin/bash
cd /root/autodl-tmp/qkn/reproduce || exit 1
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
KD=/root/autodl-tmp/qkn/outputs/kode_result
LOGF=/root/autodl-tmp/qkn/outputs/reproduce/_hp_work2.log
: > $LOGF
if [ -f "$KD/arab/knode_arab_tp1skip_all_summary_hp_act_relu.csv" ]; then
  echo "[exists] arab _hp_act_relu" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START arab _hp_act_relu" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset arab --_single act_relu --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 30 --hid 128 --depth 1 --act relu --seed 42 --tag _hp_act_relu >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   arab _hp_act_relu rc=$?" >> $LOGF
fi
if [ -f "$KD/arab/knode_arab_tp1skip_all_summary_hp_hid256.csv" ]; then
  echo "[exists] arab _hp_hid256" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START arab _hp_hid256" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset arab --_single hid256 --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 30 --hid 256 --depth 1 --act tanh --seed 42 --tag _hp_hid256 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   arab _hp_hid256 rc=$?" >> $LOGF
fi
if [ -f "$KD/rice_indica_control/knode_rice_tp1skip_all_summary_hp_act_relu.csv" ]; then
  echo "[exists] rice _hp_act_relu" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START rice _hp_act_relu" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset rice --_single act_relu --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 20 --hid 128 --depth 1 --act relu --seed 42 --tag _hp_act_relu >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   rice _hp_act_relu rc=$?" >> $LOGF
fi
if [ -f "$KD/rice_indica_control/knode_rice_tp1skip_all_summary_hp_hid256.csv" ]; then
  echo "[exists] rice _hp_hid256" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START rice _hp_hid256" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset rice --_single hid256 --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 20 --hid 256 --depth 1 --act tanh --seed 42 --tag _hp_hid256 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   rice _hp_hid256 rc=$?" >> $LOGF
fi
if [ -f "$KD/maize_rgb_drought/knode_maize_rgb_tp1skip_all_summary_hp_act_relu.csv" ]; then
  echo "[exists] uav _hp_act_relu" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START uav _hp_act_relu" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset uav --_single act_relu --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 50 --hid 128 --depth 1 --act relu --seed 42 --tag _hp_act_relu >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   uav _hp_act_relu rc=$?" >> $LOGF
fi
if [ -f "$KD/maize_rgb_drought/knode_maize_rgb_tp1skip_all_summary_hp_dim40.csv" ]; then
  echo "[exists] uav _hp_dim40" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START uav _hp_dim40" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset uav --_single dim40 --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 40 --hid 128 --depth 1 --act tanh --seed 42 --tag _hp_dim40 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   uav _hp_dim40 rc=$?" >> $LOGF
fi
echo DONE_HP_WORK2 >> $LOGF
