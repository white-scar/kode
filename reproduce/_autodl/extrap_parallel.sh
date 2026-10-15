#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
R=/root/autodl-tmp/qkn/reproduce
PY=/root/miniconda3/bin/python

# 1) ??"?? launcher"???????? maize python ???????????? arab
pkill -f 'extrap_run.sh' 2>/dev/null
echo "???? launcher; ????: $(pgrep -af knode_extrap | wc -l) ?"

# 2) ???? arab + rgb(drought) + rgb(irrigated)
cd $R || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

setsid nohup $PY knode_extrap_maize_arab_v2.py --dataset arab --n_test 5 \
  --foldsets 1,2,3,4,5,6,7,8,9,10 --epochs 800 > $L/_extrap_arab_noskip.log 2>&1 < /dev/null &
echo "arab  PID=$!"

setsid nohup $PY knode_extrap_maize_rgb_v2.py --dataset rgb --env drought --n_test 2 \
  --foldsets 1,2,3,4,5,6,7,8,9,10 --epochs 800 > $L/_extrap_rgb_drought_noskip.log 2>&1 < /dev/null &
echo "rgb-drought  PID=$!"

setsid nohup $PY knode_extrap_maize_rgb_v2.py --dataset rgb --env irrigated --n_test 2 \
  --foldsets 1,2,3,4,5,6,7,8,9,10 --epochs 800 > $L/_extrap_rgb_irrigated_noskip.log 2>&1 < /dev/null &
echo "rgb-irrigated PID=$!"

sleep 8
echo "--- ????? ---"
pgrep -af knode_extrap | sed 's/--foldsets [0-9,]*/--foldsets .../' 
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
