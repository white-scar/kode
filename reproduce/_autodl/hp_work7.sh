#!/bin/bash
cd /root/autodl-tmp/qkn/reproduce || exit 1
PY=/root/miniconda3/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
KD=/root/autodl-tmp/qkn/outputs/kode_result
LOGF=/root/autodl-tmp/qkn/outputs/reproduce/_hp_work7.log
: > $LOGF
if [ -f "$KD/arab/knode_arab_tp1skip_all_summary_hp_dim20.csv" ]; then
  echo "[exists] arab _hp_dim20" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START arab _hp_dim20" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset arab --_single dim20 --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 20 --hid 128 --depth 1 --act tanh --seed 42 --tag _hp_dim20 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   arab _hp_dim20 rc=$?" >> $LOGF
fi
if [ -f "$KD/arab/knode_arab_tp1skip_all_summary_hp_seed2025.csv" ]; then
  echo "[exists] arab _hp_seed2025" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START arab _hp_seed2025" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset arab --_single seed2025 --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 30 --hid 128 --depth 1 --act tanh --seed 2025 --tag _hp_seed2025 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   arab _hp_seed2025 rc=$?" >> $LOGF
fi
if [ -f "$KD/rice_indica_control/knode_rice_tp1skip_all_summary_hp_dim30.csv" ]; then
  echo "[exists] rice _hp_dim30" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START rice _hp_dim30" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset rice --_single dim30 --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 30 --hid 128 --depth 1 --act tanh --seed 42 --tag _hp_dim30 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   rice _hp_dim30 rc=$?" >> $LOGF
fi
if [ -f "$KD/rice_indica_control/knode_rice_tp1skip_all_summary_hp_seed2025.csv" ]; then
  echo "[exists] rice _hp_seed2025" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START rice _hp_seed2025" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset rice --_single seed2025 --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 20 --hid 128 --depth 1 --act tanh --seed 2025 --tag _hp_seed2025 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   rice _hp_seed2025 rc=$?" >> $LOGF
fi
if [ -f "$KD/maize_rgb_drought/knode_maize_rgb_tp1skip_all_summary_hp_dim100.csv" ]; then
  echo "[exists] uav _hp_dim100" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START uav _hp_dim100" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset uav --_single dim100 --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 100 --hid 128 --depth 1 --act tanh --seed 42 --tag _hp_dim100 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   uav _hp_dim100 rc=$?" >> $LOGF
fi
if [ -f "$KD/maize_rgb_drought/knode_maize_rgb_tp1skip_all_summary_hp_seed1234.csv" ]; then
  echo "[exists] uav _hp_seed1234" >> $LOGF
else
  echo "[$(date +%H:%M:%S)] START uav _hp_seed1234" >> $LOGF
  $PY knode_hyperparam_ablation.py --dataset uav --_single seed1234 --epochs 1000 --foldsets 10 --drift affine --mode ode_hyper,iter_hyper --grm-dim 50 --hid 128 --depth 1 --act tanh --seed 1234 --tag _hp_seed1234 >> $LOGF 2>&1
  echo "[$(date +%H:%M:%S)] END   uav _hp_seed1234 rc=$?" >> $LOGF
fi
echo DONE_HP_WORK7 >> $LOGF
