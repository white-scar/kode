#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
date '+?? %m-%d %H:%M:%S'
echo "=== AutoDL ?? ==="
pgrep -af knode_extrap | head -3
echo "=== maize ??? ==="
tail -5 $L/_extrap_maize_noskip.log 2>/dev/null
echo "=== arab ??? ==="
tail -5 $L/_extrap_arab_noskip.log 2>/dev/null || echo "(????, ? maize ??)"
echo "=== ??? foldset ? (???? 10) ==="
echo "  maize: $(grep -c 'foldset' $L/_extrap_maize_noskip.log 2>/dev/null) ? foldset ?"
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
