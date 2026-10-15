#!/bin/bash
cd /root/autodl-tmp/qkn || exit 9
date
echo "=== 是否还有任务在跑 ==="
pgrep -fc "R --no-echo"
pgrep -fc "knode_"
pgrep -fc "python"
echo "=== 打小文件包 ==="
tar czf /tmp/archive_small.tgz \
  outputs/kode_result/arab/*_all_summary_hp_*.csv outputs/kode_result/arab/*_all_detail_hp_*.csv outputs/kode_result/arab/*_per_foldset_hp_*.csv \
  outputs/kode_result/rice_indica_control/*_all_summary_hp_*.csv outputs/kode_result/rice_indica_control/*_all_detail_hp_*.csv outputs/kode_result/rice_indica_control/*_per_foldset_hp_*.csv \
  outputs/kode_result/maize_rgb_drought/*_all_summary_hp_*.csv outputs/kode_result/maize_rgb_drought/*_all_detail_hp_*.csv outputs/kode_result/maize_rgb_drought/*_per_foldset_hp_*.csv \
  outputs/reproduce/rank_sweep_arab_dg30/cells outputs/reproduce/rank_sweep_rice_dg10/cells outputs/reproduce/rank_sweep_uav_dg50/cells \
  outputs/reproduce/dg_rank_maize/matrix_*.csv outputs/reproduce/dg_rank_maize/agg.csv outputs/reproduce/dg_rank_maize/*.json \
  outputs/reproduce/author_tp1_rice/components/*_val_X_norm.csv \
  outputs/reproduce/author_all2_rice_innerens/components/*_enspred_*.csv \
  outputs/reproduce/knode_dynamic_baselines_full_*_v2.csv outputs/reproduce/knode_dynamic_baselines_full_*_v2.json \
  outputs/reproduce/dynamicGP_MegaLMM_arabidopsis_CV3_unified/*.csv outputs/reproduce/dynamicGP_MegaLMM_rice_CV3_unified/*.csv \
  outputs/reproduce/hyperparam_ablation_*_summary.csv 2>/dev/null
ls -la /tmp/archive_small.tgz
echo "=== 服务器上较大的可选项 ==="
du -sh outputs/reproduce/knode_dynamic_baselines_full_*_v2_preds.npz 2>/dev/null
du -sh outputs/kode_result/*/*_per_trait_hp_*.csv 2>/dev/null | tail -3
du -sh outputs/reproduce/dg_rank_maize 2>/dev/null