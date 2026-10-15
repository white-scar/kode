#!/bin/sh
cd /data/fandata/dashuaige/reproduce
PY=/data/fandata/dashuaige/miniconda3/envs/database/bin/python
OUTD=/data/fandata/dashuaige/outputs/reproduce
KD=/data/fandata/dashuaige/outputs/kode_result/arab
SNAP=/data/fandata/dashuaige/outputs/reproduce/abl_shards
mkdir -p $SNAP

CUDA_VISIBLE_DEVICES=5 nohup $PY knode_arab_np_tp1_skip_affine.py --foldsets 10 --epochs 1000 --drift linear --mode ode_hyper,ode_hyper_skip --tag _linear > $OUTD/arab_abl_l_rec.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=6 nohup $PY knode_arab_np_tp1_skip_affine.py --foldsets 10 --epochs 1000 --drift linear --mode iter_hyper,iter_hyper_skip --tag _linear > $OUTD/arab_abl_l_iter.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=7 nohup $PY knode_arab_np_tp1_skip_affine.py --foldsets 10 --epochs 1000 --drift affine --mode ode_hyper,ode_hyper_skip --tag _affine > $OUTD/arab_abl_a_rec.log 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=1 nohup $PY knode_arab_np_tp1_skip_affine.py --foldsets 10 --epochs 1000 --drift affine --mode iter_hyper,iter_hyper_skip --tag _affine > $OUTD/arab_abl_a_iter.log 2>&1 &
P4=$!
echo "ARAB_PIDS $P1 $P2 $P3 $P4"

snap() {
  pid=$1; name=$2
  while kill -0 "$pid" 2>/dev/null; do sleep 15; done
  for tag in linear affine; do
    cp "$KD/knode_arab_tp1skip_all_summary_${tag}.csv" "$SNAP/summary_${tag}_arab_${name}.csv" 2>/dev/null
    cp "$KD/knode_arab_tp1skip_all_detail_${tag}.csv"  "$SNAP/detail_${tag}_arab_${name}.csv" 2>/dev/null
  done
  echo "snapshot arab $name $(date '+%H:%M:%S')" >> $SNAP/snap.log
}
snap $P1 l_rec &
snap $P2 l_iter &
snap $P3 a_rec &
snap $P4 a_iter &
wait
echo ARAB_ALL_DONE >> $SNAP/snap.log
