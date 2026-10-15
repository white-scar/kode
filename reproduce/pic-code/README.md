# 图片 / 表格生成记录（现行版）

> 位置：`D:\pythonproject\knode\reproduce\pic-code\README.md`
> 用途：说清**每张图/表的数据链路** —— 原始脚本在哪、原始结果在哪、把原始结果规整成图用数据的脚本与文件、以及最终画图的脚本与输出。
> 另含 `related\_document\` 原始脚本归档索引（见 **§1B**）；该目录只存 ① 上游脚本副本、不可直接运行。
> 原则：**重跑只按本文档走，不要凭记忆。** 每个图目录里的 `make_*.py`（或 `.R`）= 规整脚本，`fig*.py`（或 `.R`）= 画图脚本。

---

## 0. 全局口径（所有图共用）

- **归一化**：scaler 只在**训练折 folds 1–5** 上拟合（每性状 min/max 取训练折内 基因型×时间 极值），套用到验证/测试；验证/测试不参与任何统计量。
- **集成**：inner 5 折（4/5 训练 + 1/5 早停），test 预测 = 5 折平均。
- **十折**：10 个 fold-set，fold 1–5 训练 / fold 6 测试。
- **作者 DMD**：必须 `scale="none"`（作者原版默认 `min.max` 是 per-subset，会泄漏验证集表型范围）。
- **MSE**：所有方法在同一（训练折 per-trait min-max）归一化空间内计算，可直接比较。
- **阴影 / 误差线口径**：
  - 图2 / 图7 面板 b 的阴影 = **跨性状 SD**（每个性状先跨 fold 平均 → 再取性状间 SD）；
  - 图5 面板 b 的阴影 = 跨性状 SD；图5/图7 面板 d 的误差线 = **fold-set 间 SD**（n=10）。
- **统计**：以 fold-set 为独立重复单元（n=10），percentile bootstrap 10 000 次，逐性状比较后做 Benjamini–Hochberg FDR；脚本 `reproduce\stats_paired_bootstrap_fdr.py`。
- **图片格式**：只输出 **SVG**。

---

## 1. 总览：图 → 目录 → 原始脚本 → 规整脚本 → 画图脚本

| 图 / 表 | 目录 | 原始脚本（上游） | 规整脚本 | 画图脚本 | 输出 |
|---|---|---|---|---|---|
| **图2 玉米/拟南芥/水稻** | `pic2\`、`pic2\arab-E1\`、`pic2\rice-E2\` | 见 §2 | `unified_norm_eval*.py` | `fig2_*_pcc.py` | `fig2_*_pcc*.svg` |
| **图3 条件阶梯（4 数据集）** | `pic3\`（+ `arab\ rice\ rgb\`） | `knode_cond_ladder_*.py` | `make_fig_cond_ladder*_data.py` | `fig_cond_ladder_*_v2.py` | `fig_cond_ladder_*_v2.svg` |
| **图4 rank（4 数据集）** | `pic4\`（+ `arab\ rice\ rgb\`） | `knode_rank_genetics_value*.py` | `make_fig_rank_data.py` | `fig_rank*.py` | `fig_rank*.svg` |
| **图5 动态基线（4 数据集）** | `pic5\`（+ `arab\ rice\ uav\`） | Kode per_trait + `knode_dynamic_baselines_full*.py` | `make_fig_baselines_data.py` | `fig_baselines*.py` | `fig_baselines*.svg` |
| **图6 组件+超参消融（4 数据集）** | `pic6\`（+ `arab\ rice\ uav\`） | `knode_hyperparam_ablation.py`；组件消融 run | `make_component_ablation_from_summary.py`、`make_hyperparam_data.py` | `fig_ablation_dual_*_v2.py`（UAV 为 `_v3`） | `Figure_ablation_dual_*_v2.svg` |
| **图7 时间外推（4 数据集）** | `pic7\`（+ `arab\ rice\ uav\`） | `knode_extrap_*_v2.py`；作者 `dynamicGP_*_CV3*.R` | `make_fig7_data.py`、`make_fig7_data_v2.py` | `fig7_extrap.py`（玉米）+ 各子目录 `fig7_extrap_*.py` | `fig7_extrap_*.svg` |
| **图8 无人机 RGB** | `pic8\` | `knode_maize_rgb_drought_np_tp1_affine.py` | `make_rgb_eval_v2.py` | `fig8_maize_rgb_pcc_mse_v3.py` | `fig8_maize_rgb_pcc_mse_v3.svg` |
| **补充图 E3 / E4** | `E3-4\`（+ `maize\ arab\ rice\`） | Kode per_trait + RR-BLUP benchmark | `make_table2_data.py` | `E3.R` / `E4.R` | `FigureE3_*.svg` / `FigureE4_*.svg` |
| **拓展表 E-table1** | `E-table1\` | `knode_rank_genetics_value*.py`、`knode_{arab,rice,rgb}_dg_rank_joint.py` | `make_E-table1_rank.py`、`make_E-table1_dgxrank{,_data}.py` | —（表，无画图脚本） | `E-table1_rank_paste.md`、`E-table1_rank_table.tex` |
| **表1 算子 A** | `table1\` | 服务器 author_tp1 per-fold 导出（副本见 §1B） | `make_table1_data.py` | `make_table1.py` | `table1.csv` |
| ~~表2~~ | `table2\` | **已废弃**（被 `E3-4\` 取代，目录为空） | — | — | — |

---

## 1B. 原始脚本归档（`related\_document\`）

> `related\_document\` = 把下面 §2–§11 各图/表的 **① 原始脚本（产生原始数据的那一层）** 复制一份集中归档，**只放脚本、不放结果**（结果仍在 `outputs\`，体积太大）。
> 这些副本只是**存证 / 交接用**，不是原生运行位置，**不能直接跑**（脚本里的相对路径指向仓库根，运行请用 §2–§11 给出的原始位置）。
> ③ 规整脚本（`make_*.py`）与 ⑤ 画图脚本（`fig*.py`）仍在各自图目录（`pic2\`…`pic8\`、`E3-4\`、`E-table1\`）。
> 本文件即归档的**唯一索引**；`related\_document\` 目录下不放单独的 README（原 `_document\README.md` 已删除，内容并入本节）。

### 1B.1 归档目录 → 对应图 / 表

| 子目录 | 对应图 / 表 | 归档内容 |
|---|---|---|
| `00_shared_modules\` | 多个脚本共用的数据模块 | 玉米 / 拟南芥 / 水稻 / UAV 的数据加载模块，被下面实验脚本 `import` |
| `01_main_kode_model\` | **图2 / 图5 / E3-E4** 的 Kode（本方法） | 4 数据集 `*_np_tp1_affine.py`，另含水稻 salt、UAV irrigated 两套备选环境 |
| `02_fig2_author_dynamicGP\` | **图2** | 作者 tp1 inner-5 导出脚本 + 作者 MegaLMM / RR-BLUP inner-5 复现脚本（玉米 / 拟南芥 / 水稻） |
| `03_fig3_cond_ladder\` | **图3** | 4 数据集 `knode_cond_ladder_*.py` |
| `04_fig4_rank\` | **图4** | 4 数据集 `knode_rank_genetics_value*.py` |
| `05_fig5_baselines\` | **图5** | 4 数据集 `knode_dynamic_baselines_full*.py`（ST-STP / LSTM / AR-Transformer / ODE-RNN / GRU-ODE-Style） |
| `06_fig6_ablation\` | **图6** | 超参消融包装器 + 服务器端组件消融启动 / 排队脚本 |
| `07_fig7_forecasting\` | **图7** | Kode 外推脚本（三场景）+ 作者 CV3 脚本 |
| `08_E_table1_dg_rank\` | **拓展表 E-table1** 的 D_G×rank 网格 | 4 数据集 `knode_*_dg_rank_joint.py` |
| `09_table1_operatorA\` | **表1** 算子 A 的 h² / 可预测性 | 作者侧 `author_A_innerens.py` + 本方法侧 `kode_A_sig_trainval.py` |

（合计 54 个脚本。）

### 1B.2 逐脚本来源

| 归档文件 | 原始位置 |
|---|---|
| `00_shared_modules\knode_staygreen_selection.py` | `reproduce\`（玉米数据模块） |
| `00_shared_modules\knode_perfect_vs_predicted_operators.py` | `reproduce\`（玉米数据模块，cond ladder 用） |
| `00_shared_modules\knode_arab_data.py` | `reproduce\arab\` |
| `00_shared_modules\knode_rice_data.py` | `reproduce\rice\` |
| `00_shared_modules\knode_uav_data.py` | `reproduce\` |
| `01_main_kode_model\*.py` | `reproduce\{maize_magic, arab, rice, maize_RGB}\` |
| `02_fig2_author_dynamicGP\author_all3_innerens_maize.R` | `reproduce\` |
| `02_fig2_author_dynamicGP\author_all2_innerens_rice.R` | `reproduce\` |
| `02_fig2_author_dynamicGP\author_rice_postproc.py` | `reproduce\` |
| `02_fig2_author_dynamicGP\author_tp1_{innerens,export}_{maize,arab}.R` | **服务器** `/data/fandata/dashuaige/reproduce/` |
| `02_fig2_author_dynamicGP\dynamicGP_{MegaLMM,RR}_innerens_{maize,arab,rice}.R` | `reproduce\{maize_magic, arab, rice}\` |
| `03_fig3_cond_ladder\*.py` | `reproduce\{maize_magic, arab, rice, maize_RGB}\` |
| `04_fig4_rank\*.py` | `reproduce\`（玉米）、`reproduce\{arab, rice, maize_RGB}\` |
| `05_fig5_baselines\*.py` | `reproduce\{maize_magic, arab, rice, maize_RGB}\` |
| `06_fig6_ablation\knode_hyperparam_ablation.py` | `reproduce\` |
| `06_fig6_ablation\run_{maize,arab}_ablation.sh`、`server_component_ablation_queue.py` | **服务器** |
| `07_fig7_forecasting\knode_extrap_maize_{arab,rice,rgb}_v2.py` | `reproduce\arab\`（`arab` 版同时产出玉米结果）、`reproduce\rice\`、`reproduce\` |
| `07_fig7_forecasting\dynamicGP_MegaLMM_TP1_{arabidopsis,rice,maize}_CV3*.R` | `reproduce\{arab, rice, maize_magic}\`；`..._maize_CV3_3scen.R` 在**服务器** |
| `08_E_table1_dg_rank\*.py` | `reproduce\{maize_magic, arab, rice, maize_RGB}\` |
| `09_table1_operatorA\*.py` | **服务器** |

### 1B.3 两点说明

1. **玉米的时间外推**没有单独的 `maize` 脚本：它与拟南芥共用一个脚本 `knode_extrap_maize_arab_v2.py`（脚本内按 `--dataset` 区分），所以归档在 `07_fig7_forecasting\`。
2. **表1 的 A 统计**是服务器端一次性产出的，本地原来没有脚本；已从服务器取回 `author_A_innerens.py` 与 `kode_A_sig_trainval.py` 存档。原始结果仍在 `outputs\reproduce\author_tp1_perfold\`。

---

## 2. 图2 —— 玉米 / 拟南芥 / 水稻 纵向预测（Kode vs DynamicGP 各变体）

| 环节 | 玉米（`pic2\`） | 拟南芥（`pic2\arab-E1\`） | 水稻（`pic2\rice-E2\`） |
|---|---|---|---|
| ① Kode 原始脚本 | `reproduce\maize_magic\knode_maize_np_tp1_affine.py`（10 fold-set，inner-5，affine，无 skip） | `reproduce\arab\knode_arab_np_tp1_affine.py` | `reproduce\rice\knode_rice_np_tp1_affine.py` |
| ② Kode 原始结果 | `outputs\kode_result\maize\knode_tp1_{ode_hyper,iter_hyper}_per_trait.csv` | `outputs\kode_result\arab\knode_arab_tp1_{ode_hyper,iter_hyper}_per_trait.csv` | `outputs\kode_result\rice_indica_control\knode_rice_tp1_{ode_hyper,iter_hyper}_hp_base_per_trait.csv` |
| ① 作者 tp1（inner-5 集成）原始 | 服务器 `reproduce\author_all3_innerens_maize.R`（配 `author_tp1_export_maize.R`） | 服务器 `author_all3_innerens_arab.R` | 服务器 `author_rice_postproc.py` |
| ② 作者 tp1 原始结果 | `outputs\reproduce\author_tp1_maize_innerens\components\iter*_val_{R,phi}_enspred.csv` | `outputs\reproduce\author_tp1_arab_innerens\components\...` | `pic2\rice-E2\author_rice_long_v2.csv`（服务器 postproc 产物副本） |
| ① 作者 mega / rr（inner-5）原始脚本 | `reproduce\maize_magic\dynamicGP_MegaLMM_innerens_maize.R`、`dynamicGP_RR_innerens_maize.R` | `reproduce\arab\dynamicGP_MegaLMM_innerens_arab.R`、`dynamicGP_RR_innerens_arab.R` | `reproduce\rice\dynamicGP_MegaLMM_innerens_rice.R`、`dynamicGP_RR_innerens_rice.R` |
| ② 作者 mega / rr 原始结果 | `outputs\dynamicgp_megalmm_innerens\`、`outputs\dynamicgp_rr_innerens\` | `...\dynamicgp_megalmm_arab_innerens\`、`dynamicgp_rr_innerens_arab\` | `...\dynamicgp_megalmm_rice_innerens\`、`dynamicgp_rr_innerens_rice\` |
| ③ 规整脚本 | `pic2\unified_norm_eval.py` | `pic2\arab-E1\unified_norm_eval_arab.py` | `pic2\rice-E2\unified_norm_eval_rice_v2.py` |
| ④ 规整后数据 | `pic2\unified_norm_author_vs_kode_per_timepoint.csv`、`..._per_foldset.csv` | `pic2\arab-E1\...` | `pic2\rice-E2\..._v2.csv` |
| ⑤ 画图脚本 | `pic2\fig2_maize_pcc.py` | `pic2\arab-E1\fig2_arab_pcc.py` | `pic2\rice-E2\fig2_rice_pcc_v2.py` |
| ⑥ 输出 | `pic2\fig2_maize_pcc.svg` | `pic2\arab-E1\fig2_arab_pcc.svg` | `pic2\rice-E2\fig2_rice_pcc_v2.svg` |

`unified_norm_eval*.py` 直接把规整后的 CSV 写进**脚本所在目录**（2026-09-27 起取消旧的"先写 `outputs\kode_result\` 再手拷"流程）。列：`Series, Fold, DAS, trait, PCC, MSE, R2, NMSE`。

---

## 3. 图3 —— 条件阶梯（Persistence / Shared / GRM / TP1 / Full）

| 环节 | 文件（4 个数据集：玉米 / 拟南芥 / 水稻 / 无人机） |
|---|---|
| ① 原始脚本 | `reproduce\maize_magic\knode_cond_ladder_maize.py`、`reproduce\arab\knode_cond_ladder_arab.py`、`reproduce\rice\knode_cond_ladder_rice.py`、`reproduce\maize_RGB\knode_cond_ladder_rgb.py` |
| ② 原始结果 | `outputs\reproduce\cond_ladder_maize_dg50_r2\`、`cond_ladder_arab_dg30_r2\`、`cond_ladder_rice_dg10_r2\`、`cond_ladder_rgb_dg50_r2\`（各含 `agg.csv`、`f0..f9.csv`、`trait_f0..f9.csv`） |
| ③ 规整脚本 | `pic3\make_fig_cond_ladder_data.py`（+ `arab\`、`rice\`、`rgb\` 各一份） |
| ④ 规整后数据 | `pic3\fig_cond_ladder_maize_data.csv`（+ `arab\`、`rice\`、`rgb\`） |
| ⑤ 画图脚本 + 数据 | `pic3\fig_cond_ladder_maize_v2.py`（+ `arab\fig_cond_ladder_arab_v2.py`、`rice\...`、`rgb\...`） |
| ⑥ 输出 | `pic3\fig_cond_ladder_maize_v2.svg`（+ 三个子目录同名 SVG） |

---

## 4. 图4 —— 有效动力学维度（rank 敏感性、A 的可预测性、谱能量）

| 环节 | 文件 |
|---|---|
| ① 原始脚本 | `reproduce\knode_rank_genetics_value.py`（玉米）、`reproduce\arab\knode_rank_genetics_value_arab.py`、`reproduce\rice\knode_rank_genetics_value_rice.py`、`reproduce\maize_RGB\knode_rank_genetics_value_uav.py` |
| ② 原始结果 | 玉米 `outputs\reproduce\knode_rank_genetics_value_10fold_agg.csv`；其余 `outputs\reproduce\rank_sweep_{arab_dg30, rice_dg10, uav_dg50}\agg.csv`（条件 × rank 的 PCC / A 可预测性） |
| ③ 规整脚本 | `pic4\make_fig_rank_data.py` |
| ④ 规整后数据 | `pic4\fig_rank_data.csv`（面板 a/b，长表 `panel, dataset, series, x, mean, sd`）、`pic4\fig_rank_energy.csv`（面板 c 谱能量） |
| ⑤ 画图脚本 | `pic4\fig_rank.py`（+ `arab\fig_rank_arab.py`、`rice\fig_rank_rice.py`、`rgb\fig_rank_rgb_drought.py`） |
| ⑥ 输出 | `pic4\fig_rank.svg`（+ 三个子目录同名 SVG，数据分别存 `*_data.csv`、`*_energy.csv`） |

---

## 5. 图5 —— Kode vs 四种代表性动态基线

| 环节 | 说明 |
|---|---|
| ① 基线原始脚本 | `reproduce\maize_magic\knode_dynamic_baselines_full.py`（玉米）、`reproduce\arab\knode_dynamic_baselines_full_arab.py`、`reproduce\rice\knode_dynamic_baselines_full_rice.py`、 |
| ② 基线原始结果 | `outputs\reproduce\knode_dynamic_baselines_full{,_arab_v2,_rice_v2}_per_{trait,timepoint}.csv` |
| ② Kode 原始结果 | 同 §2 的 Kode per_trait（**图5 的 Kode 与图2 同源**） |
| ③ 规整脚本 | `pic5\make_fig_baselines_data.py`（按数据集拆分输出到 `pic5\` 与 `arab\ rice\`） |
| ④ 规整后数据 | `pic5\fig_baselines_data.csv` + `fig_baselines_mse_long.csv`（+ 两个子目录同名文件） |
| ⑤ 画图脚本 | `pic5\fig_baselines.py`（玉米）+ `arab\fig_baselines_arab.py`、`rice\fig_baselines_rice.py` |
| ⑥ 输出 | `pic5\fig_baselines.svg` + 两个子目录同名 SVG |
| 附 | `pic5\table_dynamic_methods_overall.csv`（各方法总体数值表）、`pic5\_mktab2.py`（生成该表的脚本） |

---

## 6. 图6 —— 组件消融 + 超参敏感性

| 环节 | 说明 |
|---|---|
| ① 组件消融原始脚本 | 服务器上的组件消融 run（输出分片汇总到 `outputs\reproduce\abl_shards\summary_*.csv`、`detail_*.csv`） |
| ① 超参消融原始脚本 | `reproduce\knode_hyperparam_ablation.py`（包装 4 个主模型脚本，逐配置跑 1000 epoch） |
| ② 超参原始结果 | `outputs\reproduce\hyperparam_ablation_maize_current.csv`、`hyperparam_ablation_{arab,rice,uav}_summary.csv` |
| ③ 规整脚本 | `pic6\make_component_ablation_from_summary.py`（组件）、`pic6\make_hyperparam_data.py`（超参） |
| ④ 规整后数据 | `pic6\component_ablation.csv` + `pic6\hyperparam_ablation.csv`（+ `arab\`、`rice\`、`uav\`；拟南芥的组件表名为 `arab\figure9_arab_component_ablation.csv`） |
| ⑤ 画图脚本 | `pic6\fig_ablation_dual_v2.py`、`arab\fig_ablation_dual_arab_v2.py`、`rice\fig_ablation_dual_rice_v2.py`、`uav\fig_ablation_dual_uav_v3.py` |
| ⑥ 输出 | `pic6\Figure_ablation_dual_v2.svg`、`arab\Figure_ablation_dual_arab_v2.svg`、`rice\..._rice_v2.svg`、`uav\..._uav_v3.svg`（面板 a–e；f Random seed 已删除） |

---

## 7. 图7 —— 观测窗口之外的时间外推（forecasting）

| 环节 | 说明 |
|---|---|
| ① Kode 外推原始脚本 | `reproduce\arab\knode_extrap_maize_arab_v2.py`、`reproduce\rice\knode_extrap_maize_rice_v2.py`、`reproduce\knode_extrap_maize_rgb_v2.py`（玉米用同名 maize 版；三场景 rec_t1 / rec_tk / iter，输出全轨迹） |
| ② Kode 外推原始结果 | `outputs\reproduce\knode_extrap_{maize_nt5, arab_nt2, rice_nt2, rgb_nt2}\`（含 `*_overall.csv`、`*_fulltraj_*.csv`、`*_fulltraj_bytrait_*.csv` 等） |
| ① 作者 CV3 原始脚本 | `reproduce\arab\dynamicGP_MegaLMM_TP1_arabidopsis_CV3.R`、`reproduce\rice\dynamicGP_MegaLMM_TP1_rice_CV3.R`；玉米为服务器 `dynamicGP_MegaLMM_TP1_maize_CV3_3scen.R` |
| ② 作者 CV3 原始结果 | `outputs\reproduce\dynamicGP_MegaLMM_{arabidopsis,rice}_CV3_unified\snapshot_metrics_long.csv`、`outputs\reproduce\dynamicGP_MegaLMM_maize_CV3_3scen\snapshot_metrics_long.csv`（逐性状 PCC） |
| ② 作者玉米冻结值（早期参考） | `pic7\fig7b_cv3_traj_pcc_mse.csv`、`fig7c_cv3_bytrait_mse.csv`、`fig7d_overall_scenarios.csv`（只保留作者行） |
| ③ 规整脚本 | `pic7\make_fig7_data.py`（各数据集 Kode + 作者冻结值/上限并入）、`pic7\make_fig7_data_v2.py`（把新跑的作者 CV3 并入拟南芥/水稻） |
| ④ 规整后数据 | `pic7\fig7_maize_{b,c,d}.csv`；`arab\fig7_arab_{b,c,d}[_v2].csv`；`rice\fig7_rice_*`； |
| ⑤ 画图脚本 | `pic7\fig7_extrap.py`（玉米，`DS="maize"`）；`arab\fig7_extrap_arab.py`、`rice\fig7_extrap_rice.py` |
| ⑥ 输出 | `pic7\fig7_extrap_maize.svg` + 两个子目录同名 SVG |

---

## 8. 图8 —— 田间无人机 RGB（玉米干旱 / 灌溉）

| 环节 | 文件 |
|---|---|
| ① Kode 原始脚本 | `reproduce\maize_RGB\knode_maize_rgb_drought_np_tp1_affine.py`（`--env irrigated` 可跑灌溉） |
| ② Kode 原始结果 | `outputs\kode_result\maize_rgb_drought\{knode_maize_rgb_tp1_ode_hyper,iter_hyper}_per_trait.csv`（及灌溉对应文件） |
| ① 基线 | `pic8\make_rgb_eval_v2.py` 内自带 rrBLUP 基线（用 GRM 直接求解），无需外部基线 run |
| ③ 规整脚本 | `pic8\make_rgb_eval_v2.py` |
| ④ 规整后数据 | `pic8\maize_rgb_per_timepoint_v2.csv`（逐时间点 PCC/MSE±SD）、`pic8\maize_rgb_per_foldset_v2.csv`（逐 trait×fold 箱线用） |
| ⑤ 画图脚本 | `pic8\fig8_maize_rgb_pcc_mse_v3.py` |
| ⑥ 输出 | `pic8\fig8_maize_rgb_pcc_mse_v3.svg`（v3 = 只做 drought，2 面板 a/b） |

---

## 9. 补充图 E3 / E4 —— 逐性状预测精度+遗传力 / 阈值扫描

| 环节 | 文件（`E3-4\`；`<ds>` = maize / arab / rice） |
|---|---|
| ① Kode 原始结果 | 同 §2 的 Kode per_trait（与图2/图5 同源） |
| ① 其他上游 | RR-BLUP benchmark：`outputs\dynamicGP\_Training_Validation_Results\rrBLUP_benchmarks_prediction_accuracies.csv`、`outputs\dynamicGP\Arabidopsis\...`；水稻 `outputs\reproduce\author_rice_control_metrics_rr\rrBLUP_benchmarks_prediction_accuracies_rice.csv`；逐性状遗传力 `E3-4\<ds>\X_all_unrolled_h.csv` |
| ③ 规整脚本 | `E3-4\make_table2_data.py` |
| ④ 规整后数据 | `E3-4\<ds>\{figE8_<ds>_pa_bytp_v4.csv, figE8_<ds>_pa_bytrait_v4.csv, fig5_<ds>_sweep_data.csv, h2_<ds>.csv, rrblup_<ds>.csv}`（玉米/拟南芥另有 `X_all_unrolled_h.csv`） |
| ⑤ 画图脚本 | `E3-4\E3.R <ds>`（→ FigureE3）、`E3-4\E4.R <ds>`（→ FigureE4） |
| ⑥ 输出 | `E3-4\<ds>\FigureE3_<ds>.svg`、`E3-4\<ds>\FigureE4_<ds>.svg` |
| 附 | `Figure7_caption.md` / `Figure8_caption.md`（现正文当作补充图 3 / 4 使用） |

---

## 10. 拓展表 E-table1 —— operator-rank 敏感性（四数据集）

| 环节 | 文件 |
|---|---|
| ① 原始脚本 | 同 §4：`reproduce\knode_rank_genetics_value.py` 等；D_G×rank 网格由 `reproduce\{arab,rice,maize_RGB}\knode_{arab,rice,rgb}_dg_rank_joint.py`（玉米侧的 `knode_maize_dg_rank_joint.py` 未完成，故 E-table1 主表用 rank 扫描） |
| ② 原始结果 | `outputs\reproduce\knode_rank_genetics_value_10fold_agg.csv`、`rank_sweep_{arab_dg30,rice_dg10,uav_dg50}\agg.csv`；D_G×rank 网格 `outputs\reproduce\dg_rank_{arab,rice,rgb}\agg.csv` |
| ③ 规整脚本 | `E-table1\make_E-table1_rank.py`（主表：四个数据集锁定 D_G 下的 rank 扫描）；`E-table1\make_E-table1_dgxrank_data.py` + `make_E-table1_dgxrank.py`（附表：D_G×rank 网格，仅拟南芥/水稻/无人机） |
| ④ 规整后数据 | `E-table1\E-table1_rank_{long,values}.csv`、`E-table1\E-table1_dgxrank{,_data,_wide}.csv` |
| ⑤ 排版产物 | `E-table1\E-table1_rank_paste.md`（正文版，值只到 3 位小数）、`E-table1_rank_paste_altMSE4.md`（备选，MSE 4 位）、`E-table1_rank_table.tex`（booktabs 三线表） |

---

## 11. 表1 —— 算子 A 的遗传力与可预测性（玉米 / 拟南芥）

| 环节 | 文件 |
|---|---|
| ① 原始脚本 | 服务器端 author_tp1 per-fold 导出脚本（一次性产出）；副本已归档 `related\_document\09_table1_operatorA\`（`author_A_innerens.py`、`kode_A_sig_trainval.py`） |
| ② 原始结果 | `outputs\reproduce\author_tp1_perfold\{maize,arab}_{author_A_stats,author_A_innerens,kode_A_sig_trainval}.csv` |
| ③ 规整脚本 | `table1\make_table1_data.py` |
| ④ 规整后数据 | `table1\table1_data.csv`（长表 `dataset, method, metric, split, fold, value`） |
| ⑤ 汇总脚本 | `table1\make_table1.py` |
| ⑥ 输出 | `table1\table1.csv`（宽表，mean ± SD，n=10） |

⚠️ 表注必须保留：Kode 的算子 A 由模型从 [GRM, y1] 生成，其 "validation h²" 衡量的是生成算子与 GRM 的一致性，**不能**与 dynamicGP 的 validation h² 直接比较；该表以 `r_pred`（算子可预测性）为主指标。

---

## 12. 重跑顺序（按依赖）

1. **改数据/模型** → 跑 ① 原始脚本 → 得到 ② 原始结果（这是唯一需要训练的一步）。
2. 跑 ③ 规整脚本 → 得到 ④ 规整后数据（**只读②，不训练**）。
3. 跑 ⑤ 画图脚本 → 得到 ⑥ SVG（**只读④**）。

> 只改画图样式（字体/颜色/轴范围/图例）时 **只需重跑第 3 步**；
> 只改聚合口径（如 SD 定义、纳入哪些性状）时 **需重跑第 2+3 步**；
> 只有换模型/换折/换数据才需要第 1 步。
