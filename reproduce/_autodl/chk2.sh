cd /root/autodl-tmp/qkn/outputs/reproduce
echo "=== arab: conds ==="
tail -n +2 knode_rank_genetics_value_arab_f0.csv | cut -d, -f1 | sort | uniq -c
echo "=== arab: ranks ==="
tail -n +2 knode_rank_genetics_value_arab_f0.csv | cut -d, -f2 | sort -n | uniq -c