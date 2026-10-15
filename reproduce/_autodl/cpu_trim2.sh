#!/bin/bash
keep=$(pgrep -f 'cpu_work[0-3].sh' | tr '\n' ' ')
echo "??? worker shells: $keep"
killed=0
for p in $(pgrep -f 'knode_rank_genetics_value\|knode_rank_select_v2_skipaffine'); do
  ppid=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' ')
  # ??????
  gp=$(ps -o ppid= -p $ppid 2>/dev/null | tr -d ' ')
  ok=0
  for k in $keep; do
    [ "$p" = "$k" ] && ok=1
    [ "$ppid" = "$k" ] && ok=1
    [ "$gp" = "$k" ] && ok=1
  done
  if [ $ok -eq 0 ]; then
    kill $p 2>/dev/null; killed=$((killed+1))
  fi
done
echo "?? $killed ? CPU rank ??"
sleep 4
echo "?? knode ??: $(pgrep -c -f 'knode_')"
echo " ?? rank ?: $(pgrep -c -f 'knode_rank')"
echo " ?? hp ?  : $(pgrep -c -f 'knode_hyperparam')"
echo "??????: $(ps -eLf | wc -l)   loadavg: $(cat /proc/loadavg)"
