#!/bin/bash
L=/root/autodl-tmp/qkn/outputs/reproduce
echo "=== ????????????? 10 ? fold ???==="
for ds in maize arab rice rgb; do
  n=$(ls $L/knode_cond_ladder_${ds}_trait_f*.csv 2>/dev/null | wc -l)
  echo "   $ds: $n ?"
done
echo
echo "=== ????????? ==="
head -3 $L/knode_cond_ladder_rice_trait_f0.csv 2>/dev/null
wc -l $L/knode_cond_ladder_rice_trait_f0.csv 2>/dev/null
echo
echo "=== ?????4 ?????==="
for p in /root/autodl-tmp/qkn/data/maize_magic/maize_trait_BLUP_50_NArmv_traits_afterClustering.csv \
         /root/autodl-tmp/qkn/data/arabidopsis/BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv \
         /root/autodl-tmp/qkn/data/rice_salinity/rice_indica_control_traits.csv \
         /root/autodl-tmp/qkn/data/maize_rgb/maize_rgb_drought_traits.csv; do
  if [ -f "$p" ]; then printf "   OK  %-80s %.2f MB\n" "$(basename $p)" "$(stat -c%s $p | awk '{print $1/1e6}')"; else echo "   ?  $p"; fi
done
