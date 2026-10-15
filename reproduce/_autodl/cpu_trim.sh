#!/bin/bash
# ?? cpu_work0-3, ?? 4-15
for i in 4 5 6 7 8 9 10 11 12 13 14 15; do
  p=$(pgrep -f "cpu_work$i.sh")
  [ -n "$p" ] && kill $p 2>/dev/null
  for q in $(pgrep -f "knode_rank_select_v2_skipaffine\|knode_rank_genetics_value" ); do :; done
done
# ????? worker ??? python (?? CPU ?: CUDA_VISIBLE_DEVICES ??)
for p in $(pgrep -f "knode_rank_genetics_value\|knode_rank_select_v2_skipaffine"); do
  if [ -r /proc/$p/environ ] && ! tr '\0' '\n' < /proc/$p/environ | grep -q '^CUDA_VISIBLE_DEVICES=[0-9]'; then
    kill $p 2>/dev/null
  fi
done
sleep 3
echo "?? knode ??: $(pgrep -c -f 'knode_')"
echo "?? cpu_work: $(pgrep -c -f 'cpu_work')"
echo "????: $(ps -eLf | wc -l)"
