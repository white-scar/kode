#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
for f in arab rgb_drought; do
  echo "=== _extrap_${f}_noskip.log ?? ==="
  tail -5 $L/_extrap_${f}_noskip.log 2>/dev/null
  echo "   ????: $(grep -c 'foldset ' $L/_extrap_${f}_noskip.log 2>/dev/null)  (?? 3 ?)"
  echo
done
echo "=== ???????? ==="
ls -la $L/knode_extrap_maize_arab/*.csv 2>/dev/null | tail -4
ls -la $L/knode_extrap_maize_rgb/*.csv 2>/dev/null | tail -4
