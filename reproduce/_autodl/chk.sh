cd /root/autodl-tmp/qkn/outputs/reproduce
for f in arab rice uav uav_irrigated; do
  echo -n "$f unique combos: "
  tail -n +2 knode_rank_genetics_value_${f}_f0.csv | cut -d, -f1,2 | sort -u | wc -l
done