# result-pic — 各图 / 表所用原始数据

> 本目录 = `reproduce\pic-code` 各图 / 表 **画图脚本直接读取的数据文件**，按图分文件夹存放。
> 目的：拿到某张图时，能立刻查到 **“哪张图 → 哪个画图脚本 → 读了哪个数据文件”**，不用再翻散落的中间产物。
> 画图脚本本身不在这里，仍在 `reproduce\pic-code\pic2\ … pic8\`、`E3-4\`、`E-table1\` 下（**未做任何改动**）。

---

## 0. 使用说明

- 本目录下的数据文件 **保持脚本原来的文件名**，因此画图脚本无需任何修改。
- 这里的 CSV 都是流程产出的 **汇总 / 规整结果**（逐时间点、逐性状、逐 fold 的 PCC / MSE 等），**不是**第三方的原始表型、基因型数据。
- 数据链路共三层，本目录属于第 ③④ 层（规整后数据 → 画图）：
  - ① **原始脚本**（产生原始结果那一层）：归档在 `reproduce\pic-code\related\_document\`（只存脚本，不能直接跑）。
  - ② **原始运行结果**：在仓库根 `outputs\` 下（体积大，未放入本目录）。
  - ③ **规整后数据**（= 本目录）：`make_*.py` / `unified_norm_eval*.py` 读 ② 写出。
  - ④ **画图脚本**：读本目录 → 出 SVG。

---

## 1. 图2 — 玉米 / 拟南芥 / 水稻 纵向预测（Kode vs DynamicGP 各变体）

| 数据集 | 画图脚本 | 该图读取的数据（本目录内） |
|---|---|---|
| 玉米 MAGIC | `pic-code\pic2\fig2_maize_pcc.py` | `pic2\unified_norm_author_vs_kode_per_timepoint.csv`、`pic2\unified_norm_author_vs_kode_per_foldset.csv` |
| 拟南芥 | `pic-code\pic2\arab-E1\fig2_arab_pcc.py` | `pic2\arab-E1\unified_norm_author_vs_kode_per_timepoint.csv`、`…_per_foldset.csv` |
| 水稻 | `pic-code\pic2\rice-E2\fig2_rice_pcc_v2.py` | `pic2\rice-E2\unified_norm_author_vs_kode_per_timepoint_v2.csv`、`…_per_foldset_v2.csv` |

> 规整脚本：`pic2\unified_norm_eval.py`、`pic2\arab-E1\unified_norm_eval_arab.py`、`pic2\rice-E2\unified_norm_eval_rice_v2.py`。

---

## 2. 图3 — 条件阶梯（Persistence / Shared / GRM / TP1 / Full）

| 数据集 | 画图脚本 | 该图读取的数据（本目录内） |
|---|---|---|
| 玉米 MAGIC | `pic-code\pic3\fig_cond_ladder_maize_v2.py` | `pic3\fig_cond_ladder_maize_data.csv` |
| 拟南芥 | `pic-code\pic3\arab\fig_cond_ladder_arab_v2.py` | `pic3\arab\fig_cond_ladder_arab_data.csv` |
| 水稻 | `pic-code\pic3\rice\fig_cond_ladder_rice_v2.py` | `pic3\rice\fig_cond_ladder_rice_data.csv` |
| 无人机玉米 | `pic-code\pic3\rgb\fig_cond_ladder_rgb_v2.py` | `pic3\rgb\fig_cond_ladder_rgb_data.csv` |

---

## 3. 图4 — 有效动力学维度（rank 敏感性 / A 可预测性 / 谱能量）

| 数据集 | 画图脚本 | 该图读取的数据（本目录内） |
|---|---|---|
| 玉米 MAGIC | `pic-code\pic4\fig_rank.py` | `pic4\fig_rank_data.csv` |
| 拟南芥 | `pic-code\pic4\arab\fig_rank_arab.py` | `pic4\arab\agg.csv`、`pic4\arab\fig_rank_arab_energy.csv`（中间产物 `fig_rank_arab_data.csv`） |
| 水稻 | `pic-code\pic4\rice\fig_rank_rice.py` | `pic4\rice\agg.csv`、`pic4\rice\fig_rank_rice_energy.csv`（中间产物 `fig_rank_rice_data.csv`） |
| 无人机玉米 | `pic-code\pic4\rgb\fig_rank_rgb_drought.py` | `pic4\rgb\agg.csv`、`pic4\rgb\fig_rank_rgb_drought_energy.csv`（中间产物 `fig_rank_rgb_drought_data.csv`） |

> ⚠️ 拟南芥 / 水稻 / 无人机三个脚本原本从 `outputs\reproduce\rank_sweep_<ds>\agg.csv` 读取；
> 本目录把那份 `agg.csv` 一并收进来了（放在 `pic4\<ds>\agg.csv`），方便对照，**但没有改脚本**。

---

## 4. 图5 — Kode vs 四种代表性动态基线

| 数据集 | 画图脚本 | 该图读取的数据（本目录内） |
|---|---|---|
| 玉米 MAGIC | `pic-code\pic5\fig_baselines.py` | `pic5\fig_baselines_data.csv`、`pic5\fig_baselines_mse_long.csv` |
| 拟南芥 | `pic-code\pic5\arab\fig_baselines_arab.py` | `pic5\arab\fig_baselines_data.csv`、`…_mse_long.csv` |
| 水稻 | `pic-code\pic5\rice\fig_baselines_rice.py` | `pic5\rice\fig_baselines_data.csv`、`…_mse_long.csv` |
| 无人机玉米 | `pic-code\pic5\uav\fig_baselines_uav.py` | `pic5\uav\fig_baselines_data.csv`、`…_mse_long.csv` |

> 另有总体数值表 `pic5\table_dynamic_methods_overall.csv`（由 `pic5\_mktab2.py` 生成，不在画图读取列表内）。

---

## 5. 图6 — 组件消融 + 超参敏感性

| 数据集 | 画图脚本 | 该图读取的数据（本目录内） |
|---|---|---|
| 玉米 MAGIC | `pic-code\pic6\fig_ablation_dual_v2.py` | `pic6\component_ablation.csv`、`pic6\hyperparam_ablation.csv` |
| 拟南芥 | `pic-code\pic6\arab\fig_ablation_dual_arab_v2.py` | `pic6\arab\figure9_arab_component_ablation.csv`、`pic6\arab\hyperparam_ablation_arab.csv` |
| 水稻 | `pic-code\pic6\rice\fig_ablation_dual_rice_v2.py` | `pic6\rice\component_ablation.csv`、`pic6\rice\hyperparam_ablation_rice.csv` |
| 无人机玉米 | `pic-code\pic6\uav\fig_ablation_dual_uav_v3.py` | `pic6\uav\component_ablation.csv`、`pic6\uav\hyperparam_ablation_uav.csv` |

---

## 6. 图7 — 观测窗口之外的时间外推（forecasting）

| 数据集 | 画图脚本 | 该图读取的数据（本目录内） |
|---|---|---|
| 玉米 MAGIC | `pic-code\pic7\fig7_extrap.py` | `pic7\fig7_maize_b.csv`、`fig7_maize_c.csv`、`fig7_maize_d.csv` |
| 拟南芥 | `pic-code\pic7\arab\fig7_extrap_arab.py` | `pic7\arab\fig7_arab_b_v2.csv`、`_c_v2.csv`、`_d_v2.csv` |
| 水稻 | `pic-code\pic7\rice\fig7_extrap_rice.py` | `pic7\rice\fig7_rice_b_v2.csv`、`_c_v2.csv`、`_d_v2.csv` |
| 无人机玉米 | `pic-code\pic7\uav\fig7_extrap_uav.py` | `pic7\uav\fig7_uav_b.csv`、`fig7_uav_c.csv`、`fig7_uav_d.csv` |

> `b` = 逐时间点 PCC / 轨迹，`c` = 逐性状 MSE，`d` = 三种信息条件的总体 PCC / MSE。

---

## 7. 图8 — 田间无人机 RGB（玉米干旱）

| 数据集 | 画图脚本 | 该图读取的数据（本目录内） |
|---|---|---|
| 无人机玉米（drought） | `pic-code\pic8\fig8_maize_rgb_pcc_mse_v3.py` | `pic8\maize_rgb_per_timepoint_v2.csv`、`pic8\maize_rgb_per_foldset_v2.csv` |

---

## 8. 补充图 E3 / E4 — 逐性状预测精度 + 遗传力 / 阈值扫描

| 图 | 画图脚本 | 该图读取的数据（本目录内） |
|---|---|---|
| 补充图 E3 | `pic-code\E3-4\E3.R <maize\|arab\|rice>` | `E3-4\<ds>\figE8_<ds>_pa_bytp_v4.csv`、`figE8_<ds>_pa_bytrait_v4.csv`、`h2_<ds>.csv` |
| 补充图 E4 | `pic-code\E3-4\E4.R <maize\|arab\|rice>` | `E3-4\<ds>\fig5_<ds>_sweep_data.csv` |

> `<ds>` = `maize` / `arab` / `rice`，三套分别在 `E3-4\maize\`、`E3-4\arab\`、`E3-4\rice\`。
> ⚠️ 注意：`E3.R` 除上表外，还会读取**第三方原始表型表**（未收录在本目录）：
> - maize → `data/maize_magic/maize_trait_BLUP_50_NArmv_traits_afterClustering.csv`
> - arab → `data/arabidopsis/BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv`
> - rice → `data/rice_salinity/rice_indica_control_traits.csv`

---

## 9. 拓展表 E-table1 — operator-rank 敏感性

| 内容 | 生成 / 排版脚本 | 数据（本目录内） |
|---|---|---|
| 主表（四数据集锁定 D_G 下的 rank 扫描） | `pic-code\E-table1\make_E-table1_rank.py` | `E-table1\E-table1_rank_values.csv`（主）、`E-table1_rank_wide.csv`、`E-table1_rank_long.csv` |
| 附表（D_G × rank 网格，仅拟南芥 / 水稻 / 无人机） | `pic-code\E-table1\make_E-table1_dgxrank_data.py`、`make_E-table1_dgxrank.py` | `E-table1\E-table1_dgxrank_wide.csv`、`E-table1_dgxrank_data.csv`、`E-table1_dgxrank.csv` |

> E-table1 是表格，没有画图脚本；排版产物（`.md` / `.tex`）未放入本目录。

---

## 10. 上游对照（不在本目录）

| 层 | 位置 | 说明 |
|---|---|---|
| ① 原始脚本 | `reproduce\pic-code\related\_document\` | 各图 ① 上游脚本副本（不能直接跑） |
| ② 原始运行结果 | `outputs\...` | 模型 run 的原始输出，体积大 |
| — Kode 主模型脚本 | `reproduce\{maize_magic, arab, rice, maize_RGB}\` | 产生 ② 的训练脚本 |
| — 作者方法脚本 | 同 ① / 服务器 | DynamicGP 三变体 |

---

## 11. 文件规模

- 合计 **68 个 CSV，约 37 MB**（`pic2` 和 `pic5` 的 per-foldset / per-trait 长表占大头）。
- 其余目录都是 KB 级小表。
