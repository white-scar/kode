#!/bin/bash
# _rank_smoke.sh -- 三个数据集的 rank sweep 冒烟 (1 fold x 1 cond x 1 rank x 50 ep)
set -u
cd /root/autodl-tmp/qkn || exit 9
PY=/root/miniconda3/bin/python
for kv in "arab reproduce/arab/knode_rank_genetics_value_arab.py -" \
          "rice reproduce/rice/knode_rank_genetics_value_rice.py -" \
          "uav reproduce/maize_RGB/knode_rank_genetics_value_uav.py drought"; do
  set -- $kv
  echo "===== $1 ====="
  if [ "$3" = "-" ]; then $PY "$2" --smoke 2>&1 | tail -5; else $PY "$2" --smoke --env "$3" 2>&1 | tail -5; fi
done
