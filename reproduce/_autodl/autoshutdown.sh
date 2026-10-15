#!/bin/bash
# autoshutdown.sh [??]  ?? ???????????????AutoDL ????
MIN=${1:-60}
LOG=/root/autodl-tmp/qkn/outputs/reproduce/_autoshutdown.log
DEADLINE=$(date -d "+${MIN} minutes" '+%Y-%m-%d %H:%M:%S')
{
  echo "=================================================="
  echo "[$(date '+%F %T')] ???????: ${MIN} ??? -> $DEADLINE"
  for c in shutdown poweroff halt; do
    printf "   %-9s %s\n" "$c" "$(command -v $c || echo '(???)')"
  done
} >> $LOG 2>&1

sleep $((MIN*60))

{
  echo "[$(date '+%F %T')] === ??????? ==="
  echo "[$(date '+%F %T')] ???? knode ??"
  pkill -f 'knode_' 2>/dev/null; sleep 5
  pkill -9 -f 'knode_' 2>/dev/null
  sync
  ok=0
  for cmd in "shutdown -h now" "poweroff" "halt -p" "init 0"; do
    echo "[$(date '+%F %T')] ??: $cmd"
    $cmd >> $LOG 2>&1 && { ok=1; break; }
    sleep 3
  done
  [ $ok -eq 0 ] && echo "[$(date '+%F %T')] ????????? -> ?? AutoDL ??????????"
  echo "[$(date '+%F %T')] ??"
} >> $LOG 2>&1
