#!/bin/bash
# autoshutdown2.sh "<????>" <??>  ?? ??????(????, ??????)
AT="$1"
TAG="${2:-A}"
LOG=/root/autodl-tmp/qkn/outputs/reproduce/_autoshutdown.log
target=$(date -d "$AT" +%s); now=$(date +%s); remain=$((target-now))
echo "[$(date '+%F %T')] [timer-$TAG] ?? $AT (?? ${remain}s)" >> $LOG
[ $remain -gt 0 ] && sleep $remain

{
echo "[$(date '+%F %T')] [timer-$TAG] === ?? ==="
pkill -f 'knode_' 2>/dev/null; sleep 3; pkill -9 -f 'knode_' 2>/dev/null; sync

for i in 1 2 3; do
  echo "[$(date '+%F %T')] [timer-$TAG] ? $i ?: shutdown -h now"
  shutdown -h now >> $LOG 2>&1; sleep 4
  echo "[$(date '+%F %T')] [timer-$TAG] ? $i ?: poweroff -f"
  poweroff -f >> $LOG 2>&1; sleep 4
  echo "[$(date '+%F %T')] [timer-$TAG] ? $i ?: halt -f"
  halt -f >> $LOG 2>&1; sleep 4
done

echo "[$(date '+%F %T')] [timer-$TAG] ????: ??????? (kill -9 1)"
for i in 1 2 3; do
  kill -9 1 2>> $LOG
  sleep 5
done
echo "[$(date '+%F %T')] [timer-$TAG] ???? -> ?? AutoDL ??????????"
} >> $LOG 2>&1
