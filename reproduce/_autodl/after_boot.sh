#!/bin/bash
L=/root/autodl-tmp/qkn/outputs
echo "=== ?? ==="; date '+%F %T'
echo "=== ??????? ==="; pgrep -af "knode_" | wc -l
echo
echo "=== ?7 ????? / ????==="
ls -la $L/reproduce/knode_extrap_maize_arab/ 2>/dev/null | grep -E "v2_f10_nt5(_summary)?\.csv" | awk '{print "   "$5" B  "$6" "$7" "$8"  "$9}'
echo "=== ?8 ???RGB ?? / ???==="
ls -la $L/reproduce/knode_extrap_maize_rgb/ 2>/dev/null | grep -E "v2_f10_nt2(_summary)?\.csv" | awk '{print "   "$5" B  "$6" "$7" "$8"  "$9}'
echo
echo "=== ?????????????????==="
tail -12 $L/reproduce/_autoshutdown.log 2>/dev/null
echo
echo "=== ?? ==="
df -h /root/autodl-tmp | tail -1
