#!/bin/bash
cd /root/autodl-tmp/qkn/reproduce || exit 1
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
KD=/root/autodl-tmp/qkn/outputs/kode_result
LOGF=/root/autodl-tmp/qkn/outputs/reproduce/_hp_work0.log
: > $LOGF
if [ -f "$KD/arab/knode_arab_tp1skip_all_summary_hp_act_gelu.csv" ]; then
  echo "[exists] arab _hp_act_gelu" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START arab _hp_act_gelu" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset arab --_single act_gelu --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 30 --hid 128 --depth 1 --act gelu --seed 42 --tag _hp_act_gelu >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   arab _hp_act_gelu rc=$?" >> $LOGF
fi
if [ -f "$KD/arab/knode_arab_tp1skip_all_summary_hp_dim40.csv" ]; then
  echo "[exists] arab _hp_dim40" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START arab _hp_dim40" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset arab --_single dim40 --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 40 --hid 128 --depth 1 --act tanh --seed 42 --tag _hp_dim40 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   arab _hp_dim40 rc=$?" >> $LOGF
fi
if [ -f "$KD/rice_indica_control/knode_rice_tp1skip_all_summary_hp_act_gelu.csv" ]; then
  echo "[exists] rice _hp_act_gelu" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START rice _hp_act_gelu" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset rice --_single act_gelu --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 20 --hid 128 --depth 1 --act gelu --seed 42 --tag _hp_act_gelu >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   rice _hp_act_gelu rc=$?" >> $LOGF
fi
if [ -f "$KD/rice_indica_control/knode_rice_tp1skip_all_summary_hp_dim40.csv" ]; then
  echo "[exists] rice _hp_dim40" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START rice _hp_dim40" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset rice --_single dim40 --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 40 --hid 128 --depth 1 --act tanh --seed 42 --tag _hp_dim40 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   rice _hp_dim40 rc=$?" >> $LOGF
fi
if [ -f "$KD/maize_rgb_drought/knode_maize_rgb_tp1skip_all_summary_hp_act_gelu.csv" ]; then
  echo "[exists] uav _hp_act_gelu" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START uav _hp_act_gelu" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset uav --_single act_gelu --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 50 --hid 128 --depth 1 --act gelu --seed 42 --tag _hp_act_gelu >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   uav _hp_act_gelu rc=$?" >> $LOGF
fi
if [ -f "$KD/maize_rgb_drought/knode_maize_rgb_tp1skip_all_summary_hp_dim20.csv" ]; then
  echo "[exists] uav _hp_dim20" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START uav _hp_dim20" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset uav --_single dim20 --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 20 --hid 128 --depth 1 --act tanh --seed 42 --tag _hp_dim20 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   uav _hp_dim20 rc=$?" >> $LOGF
fi
if [ -f "$KD/maize_rgb_drought/knode_maize_rgb_tp1skip_all_summary_hp_seed2024.csv" ]; then
  echo "[exists] uav _hp_seed2024" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START uav _hp_seed2024" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset uav --_single seed2024 --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 50 --hid 128 --depth 1 --act tanh --seed 2024 --tag _hp_seed2024 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   uav _hp_seed2024 rc=$?" >> $LOGF
fi
echo DONE_HP_WORK0 >> $LOGF
