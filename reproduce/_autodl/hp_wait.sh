#!/bin/bash
sleep 120
bash /root/autodl-tmp/qkn/reproduce/_autodl/hp_status.sh
echo "=== worker ???? ==="
for i in 0 1 2 3; do
  echo "-- work$i"
  tail -4 /root/autodl-tmp/qkn/outputs/reproduce/_hp_work$i.log
done
