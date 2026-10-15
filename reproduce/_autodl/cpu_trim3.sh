#!/bin/bash
KEEP="6839 6840 6841 6842"
before=$(pgrep -c -f knode_rank)
for p in $(pgrep -f knode_rank_genetics_value) $(pgrep -f knode_rank_select_v2_skipaffine); do
  pp=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' ')
  case " $KEEP " in
    *" $pp "*) echo "  ?? pid=$p (parent=$pp)";;
    *) kill $p 2>/dev/null; echo "  ?? pid=$p (parent=$pp)";;
  esac
done
sleep 4
echo "rank ??: $before -> $(pgrep -c -f knode_rank)"
echo "knode ???: $(pgrep -c -f 'knode_')  (hp ?? 8)"
echo "loadavg: $(cat /proc/loadavg)"
