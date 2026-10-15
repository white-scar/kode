#!/bin/bash
AT="2026-09-17 02:18:48"
D=/root/autodl-tmp/qkn/reproduce/_autodl
pkill -f autoshutdown.sh 2>/dev/null
pkill -f autoshutdown2.sh 2>/dev/null
sleep 1
cd $D || exit 1
setsid nohup bash autoshutdown2.sh "$AT" A >/dev/null 2>&1 </dev/null &
sleep 1
setsid nohup bash autoshutdown2.sh "$AT" B >/dev/null 2>&1 </dev/null &
sleep 4
echo "=== ????? ==="
pgrep -af autoshutdown2.sh
echo "=== ?? ==="
tail -4 /root/autodl-tmp/qkn/outputs/reproduce/_autoshutdown.log
echo "=== ?? / ?? ==="
date '+?? %F %T'
echo "??  $AT"
echo "=== sysrq ?? ==="
ls -l /proc/sysrq-trigger
echo "sysrq kernel value: $(cat /proc/sys/kernel/sysrq)"
