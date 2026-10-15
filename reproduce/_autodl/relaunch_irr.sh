#!/bin/bash
cd /root/autodl-tmp/qkn/reproduce || exit 1
if pgrep -f "env irrigated" > /dev/null; then
  echo "irrigated ???"
else
  setsid nohup /root/miniconda3/bin/python knode_extrap_maize_rgb_v2.py --dataset rgb --env irrigated \
    --n_test 2 --foldsets 1,2,3,4,5,6,7,8,9,10 --epochs 800 \
    > /root/autodl-tmp/qkn/outputs/reproduce/_extrap_rgb_irrigated_noskip.log 2>&1 < /dev/null &
  echo "??? irrigated PID=$!"
fi
sleep 6
echo "?????: $(pgrep -f knode_extrap | wc -l)"
pgrep -af knode_extrap | sed -E 's#/root/miniconda3/bin/python ##' | sed -E 's/--foldsets [0-9,]+/.../'
