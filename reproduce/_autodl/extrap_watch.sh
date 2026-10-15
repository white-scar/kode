#!/bin/bash
sleep 120
L=/root/autodl-tmp/qkn/outputs/reproduce
echo "=== ?? ==="; pgrep -af knode_extrap | head -3
echo "=== ??? ==="; tail -4 $L/_extrap_maize_noskip.log
echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== ???? foldset ?? ==="; grep -c "foldset" $L/_extrap_maize_noskip.log 2>/dev/null
date '+%H:%M:%S'
