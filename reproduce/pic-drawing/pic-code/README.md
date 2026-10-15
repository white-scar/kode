# 图片 / 表格生成记录

> 位置：`D:\pythonproject\knode\reproduce\pic-code\README.md`
> 内容：全局规范（字体字号 / 输出格式 / 共用口径）+ 每张图的上游→数据→绘图→输出 + 重跑顺序。
> 原则：**重跑只按本文档走，不要凭记忆。**

---

## 0. 全局规范

### 0.1 字体字号（所有图统一，已用 SVG 复核）

| 元素 | 字号 | 字重 |
|---|---|---|
| 面板标签 `a`/`b`/`c`/`d` | **20** | bold |
| 面板标题 | **14** | bold |
| 轴标题（xlabel / ylabel） | **12** | bold |
| 图例 | **10.5** | bold |
| 数值标注（柱顶数字、ΔPCC、百分比等） | **10.5** | bold |
| 坐标刻度数字 | **9** | bold |

字体族：**Times New Roman**（fallback `DejaVu Serif`）。全图字重统一 **700 (bold)**。

标准 rcParams（每个画图脚本开头都放这段）：

```python
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10.5,
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "savefig.dpi": 300, "figure.dpi": 110, "svg.fonttype": "none",
})
```

**五个必须避开的坑：**

1. **不要在 rcParams 里写 `"font.weight": "bold"`。**
   它和显式 `fontweight="bold"` 同时用会触发
   `findfont: Failed to find font weight bold, now using 400`，
   部分文字**会悄悄退成常规体**。要加粗就逐个元素显式写 `fontweight="bold"`。

2. **mathtext 必须设字体。**
   否则 `$\Delta$`、`$10^{-2}$`、`$\log_{10}$`、`$h^2$` 会 fallback 成 `DejaVu Sans`，
   和正文的 Times New Roman 不一致。必须保留上面 `mathtext.*` 那 4 行。

3. **面板标签要独立成 20 pt 文本**，不要塞进标题字符串（如 `"(a) Maize MAGIC..."`）。
   塞进标题后字号会跟着标题走（14 pt），和其它图不一致。

4. **刻度要单独设**：`ax.tick_params(labelsize=9)` + 循环 `t.set_fontweight("bold")`；
   柱/箱下方的**类别名**（如 Persistence / Shared）属"标签"而非刻度数字，用 **10.5**。

5. **只输出 SVG**：`for ext in ("svg",):`。不要 PNG / PDF。

**字体复核命令**（改完图跑一下，确认只有 Times New Roman + 700）：

```python
import io, re
s = io.open(r"…\xxx.svg", encoding="utf-8", errors="replace").read()
print(sorted(set(re.findall(r"font-family:\s*([^;\"']+)", s))))
print(sorted(set(re.findall(r"font-weight:\s*([^;\"']+)", s))))
print(sorted(set(float(x) for x in re.findall(r"font-size:\s*([0-9.]+)px", s))))
# 期望：families 只有 'Times New Roman'；weights = ['700']；
#       sizes 是 9 / 10.5 / 12 / 14 / 20 的子集（mathtext 上标会自动变小，属正常）
```

### 0.2 共用口径（2026-09-15 更新）

- **归一化**：scaler 只在**训练折 folds 1–5** 上拟合（每性状 min/max 取训练折内 基因型×时间 极值），套到验证集；验证/测试集不参与任何统计量。
- **集成**：inner 5 折（4/5 训练 + 1/5 早停），test 预测取 5 折平均。
- **十折**：10 个 fold-set，fold 1–5 训练 / fold 6 测试。
- **作者 DMD**：必须 `scale = "none"`（作者原版默认 `min.max` 是 per-subset，会泄漏验证集）。
- **方法变体（重要）**：Kode 现行主口径 = **affine、无 skip**
  - 玉米主图（图2 玉米 / 图5 玉米 / 表2 玉米）= `knode_tp1skip_{ode_hyper,iter_hyper}_per_trait.csv`（9/3 run，affine 无 skip）
  - `*_{ode,iter}_hyper_skip` 档**只出现在组件消融（pic6）里作对照**，正文图不再使用
  - 拟南芥 / 水稻 / 无人机 RGB 的主图**目前仍是 affine + skip**（`*_hyper_skip`）；无 skip 的结果**已存在**（同一次 run 产出），需要统一时只换读数 + 重画，无需重跑
- **SD 口径（已改）**：图上阴影带 = **跨性状 SD**（每个性状先跨折平均 → 再取性状间标准差），由各绘图脚本内的 `trait_sd()` 现算；CSV 里的 `_sd`（pooled，跨性状×折）与 `_sd_fold`（折级）两列**仅作备用**。
- **统计口径**：以 **fold-set 为独立重复单元（n = 10）**；percentile bootstrap **10 000 次**；双侧 p = 2×min(P(mean*≤0), P(mean*≥0))；逐性状比较后再做 **Benjamini–Hochberg** FDR。脚本：`reproduce\stats_paired_bootstrap_fdr.py`。

### 0.3 图 → 文件夹 → 上游 总览（现行正统）

| 图 | 文件夹 | 主上游结果 | 绘图脚本 |
|---|---|---|---|
| 图2 玉米 / 拟南芥 / 水稻 | `pic2\` + `arab-E1\` + `rice-E2\` | Kode `outputs\kode_result\{maize,arab,rice_indica_control}\`；作者 `author_tp1_*_innerens`、`dynamicgp_megalmm{,_arab,_rice}_innerens` + `dynamicgp_rr_innerens{,_arab,_rice}` | `fig2_{maize,arab,rice}_pcc.py` |
| 图3 条件阶梯 | `pic3\` | `outputs\reproduce\knode_cond_ladder_maize_*` | `fig_cond_ladder_maize.py` |
| 图4 rank | `pic4\` | `knode_rank_genetics_value*`、`rank_select_v2_skipaffine*`、`uv2\*.npz` | `fig_rank.py` |
| 图5 基线（玉米/拟南芥/水稻） | `pic5\` | Kode per_trait + `knode_dynamic_baselines_full{,_arab,_rice}_per_trait.csv` | `fig_baselines{,_arab,_rice}.py` |
| 图6 组件+超参消融（玉米 / 拟南芥） | `pic6\` + `pic6\arab\` | `outputs\reproduce\abl_shards\summary_*.csv`；`knode_tp1skip_*_hp_*_per_trait.csv` | `fig_ablation_dual.py`；`arab\pic9_arab_plot.py` |
| 图7 外推（玉米） | `pic7\` | `pic7\fig7b/7c/7d_*.csv` | `fig7_extrap_maize.py` |
| 图8 无人机 RGB | `pic8\` | `outputs\kode_result\maize_rgb_{drought,irrigated}\` | `fig8_maize_rgb_pcc_mse.py` |
| 图9 缺失值（玉米/拟南芥） | `pic9\` | `outputs\reproduce\knode_missing_maize_arab\`、`missing_per_timepoint\` | `new_fig_missing*.py` |
| 图10 水稻 in-window（旧版，待定） | `pic10\` | `pic10\kode_rice_detail.csv` 等 | `fig_rice_inwindow.py` |
| 表1 算子 A 遗传力/可预测性 | `table1\` | `outputs\reproduce\author_tp1_perfold\` | `make_table1{,_data}.py` |
| 表2 图7/图8（三数据集） | `table2\maize|arab|rice\` | pic2 的 unified 长表 + `rrBLUP_benchmarks_*` + 本地 `X_all_unrolled_h.csv` | `fig7_table2.R` / `fig8_table2.R` |

---

## 图二 —— 玉米 PCC / MSE　→ `pic2\`

### ① 上游

| 方法 | 方法脚本 | 中间结果 |
|---|---|---|
| **Kode+TP1** | `reproduce\knode_maize_np_tp1_affine.py --foldsets 10`（默认 `--drift affine`） | `outputs\kode_result\maize\knode_tp1skip_{ode_hyper,iter_hyper}_per_trait.csv`（无 skip） |
| **DynamicGP-MegaLMM+TP1** | 服务器 `reproduce\author_tp1_innerens_maize.R`（配合 `author_tp1_export_maize.R`） | `outputs\reproduce\author_tp1_maize_innerens\components\`（20 个 R/φ 集成预测） |
| **DynamicGP-MegaLMM** / **RR-BLUP** | `reproduce\dynamicGP_MegaLMM_innerens_{maize,arab,rice}.R` / `dynamicGP_RR_innerens_{maize,arab,rice}.R`????????inner-5 ??? | `outputs\dynamicgp_megalmm_innerens\` + `outputs\dynamicgp_rr_innerens\` |

### ② 图专用数据

```
reproduce\pic-code\pic2\unified_norm_eval.py   ?2026-09-27 ?????? pic2\?????"?? kode_result ???"???
  ? pic2\unified_norm_author_vs_kode_per_{timepoint,foldset}.csv?????????
```
列：`Series, Fold, DAS, trait, PCC, MSE, R2, NMSE`（2026-09-15 起新增 R²/NMSE；PCC/MSE 口径未变）。

### ③ 画图

```
reproduce\pic-code\pic2\fig2_maize_pcc.py  →  fig2_maize_pcc.svg
```
4 面板：rec PCC / iter PCC / rec log10(MSE) 箱线 / iter log10(MSE) 箱线；x 轴真实 DAS（16–47），四面板刻度对齐。图例我们方法写 `Kode`（不带 +TP1）。

---

## 图二 —— 拟南芥（补充图1）　→ `pic2\arab-E1\`

上游同玉米，脚本换成 `knode_arab_np_tp1_skip_affine.py`（**当前图仍读 `*_hyper_skip`**）、`author_tp1_arab_innerens\components\`、`outputs\dynamicgp_megalmm_arab_innerens\` + `outputs\dynamicgp_rr_innerens_arab\`。
数据与绘图：`pic2\arab-E1\unified_norm_eval_arab.py → unified_norm_author_vs_kode_per_*.csv`；`fig2_arab_pcc.py → fig2_arab_pcc.svg`（同为 4 面板）。

---

## 图二 —— 水稻（补充图2）　→ `pic2\rice-E2\`

上游：`outputs\kode_result\rice_indica_control\knode_rice_tp1skip_{ode_hyper_skip,iter_hyper_skip}rice_full10_per_trait.csv`（**skip 口径**）+ `author_rice_long_v2.csv`（服务器 `author_rice_postproc.py` 产物）。
??????`pic2\rice-E2\unified_norm_eval_rice_v2.py`?`fig2_rice_pcc_v2.py ? fig2_rice_pcc_v2.svg`?**?**????Kode / DynamicGP-MegaLMM+TP1 / DynamicGP-MegaLMM / **DynamicGP-RR-BLUP**?????? inner-5 ???? `dynamicGP_MegaLMM_innerens_rice.R`?`dynamicGP_RR_innerens_rice.R`??

---

## 图三 —— 条件信息阶梯　→ `pic3\`

- 上游：`reproduce\knode_cond_ladder_maize.py`（服务器跑；依赖 `knode_perfect_vs_predicted_operators.py` 作数据模块）→ `outputs\reproduce\knode_cond_ladder_maize_{f0..f9,trait_f0..f9,agg}.csv`
- 口径：10 fold-set、inner-5、训练折归一化、rank=2、hid=128、**affine + skip**
- 数据：`pic3\make_fig_cond_ladder_data.py → fig_cond_ladder_maize_data.csv`（`block=overall` 5 行 / `block=trait` 250 行）
  - ⚠️ **`full` 一档已替换为“我们的主结果”**（2026-09-14）：该行数值不再取阶梯脚本自带 run，而是从 `outputs\kode_result\maize\knode_tp1skip_ode_hyper_per_trait.csv` 按同一公式重算（affine 无 skip）。其余四档未动。
- 绘图：`pic3\fig_cond_ladder_maize.py → fig_cond_ladder_maize.svg`（4 面板）

---

## 图四 —— rank 分析　→ `pic4\`

| 用途 | 脚本 | 中间结果 | 变体 |
|---|---|---|---|
| panel a/b：PCC、h²、可预测性 vs rank | `knode_rank_genetics_value.py`（`_arab.py`） | `outputs\reproduce\knode_rank_genetics_value{,_arab}_10fold_agg.csv` | **linear、无 skip**（待统一到 affine） |
| panel c：累积谱能量 | `knode_rank_select_v2_skipaffine.py`（`_arab.py`） | `outputs\reproduce\rank_select_v2_skipaffine_f0..f9.csv` | **affine + skip**（脚本注明 c_i 与 skip 始终开启，秩分析不受影响） |
| panel c 四条件（含 const） | 服务器 `outputs\reproduce\uv2\gen_*_r50_f0..f9.npz` 现算 | 已固化 `fig_rank_energy.csv` | — |

- rank 网格：`1,2,3,4,6,8,10,16,20,30,50`（拟南芥末档 45）
- 数据：`pic4\make_fig_rank_data.py → fig_rank_data.csv / fig_rank_energy.csv`
- 绘图：`pic4\fig_rank.py → fig_rank.svg`（a 轨迹 PCC、b 算子 A 的 SNP 可预测性、c 累积谱能量；`EXCLUDE = {"b": {"shared"}}`）

---

## 图 5 —— 动态基线　→ `pic5\`

- 上游：Kode per_trait（玉米无 skip；拟南芥/水稻 skip）+ 5 个动态基线（ST-STP / LSTM / AR-Transformer / ODE-RNN / GRU-ODE），后者由 `knode_dynamic_baselines_full{,_arab,_rice}.py` 产出 → `knode_dynamic_baselines_full{,_arab,_rice}_per_trait.csv`（列含 `pcc, mse, R2, NMSE, RMSE, NRMSE`）
- 数据：`pic5\make_fig_baselines_data.py → fig_baselines_data.csv`（逐时间点均值 + **跨性状 SD**）/ `fig_baselines_mse_long.csv`（逐 trait×折，箱线用）
- 绘图：`fig_baselines.py`（玉米）/ `fig_baselines_arab.py` / `fig_baselines_rice.py` → 三张 SVG

---

## 图 6 —— 组件与超参消融　→ `pic6\`（玉米）+ `pic6\arab\`（拟南芥）

- 上游：组件消融（`--drift linear|affine` × 4 模式，10 fold-set、1000 epoch）→ 服务器 `outputs\reproduce\abl_shards\summary_{linear,affine}_{l_rec,l_iter,a_rec,a_iter}.csv`（分片快照，因同 tag 会互相覆盖）
- 数据：`pic6\make_component_ablation_from_summary.py --dataset maize|arab --indir outputs\reproduce\abl_shards`
  → `pic6\component_ablation.csv`（玉米，linear vs affine）与 `pic6\arab\figure9_arab_component_ablation.csv`
- 超参：`pic6\hyperparam_ablation.csv`（玉米 19 配置，来自服务器 `knode_tp1skip_*_hp_*` 汇总）
- 绘图：`pic6\fig_ablation_dual.py → Figure_ablation_dual.svg`（2×3：组件 + 5 个超参网格，MSE 右轴 0–0.010）；`pic6\arab\pic9_arab_plot.py → Figure9_arab_component_ablation.svg`

---

## 图 7 —— 外推（玉米）　→ `pic7\`

- 数据：`pic7\fig7b_cv3_traj_pcc_mse.csv`（逐 DAS mean/sd）、`fig7c_cv3_bytrait_mse.csv`（逐 trait MSE）、`fig7d_overall_scenarios.csv`（三条件总体）
- 绘图：`pic7\fig7_extrap_maize.py → fig7_extrap_maize.svg`（a 协议示意 / b PCC / c MSE 箱线 / d 三条件对照）
  - ⚠️ 面板字母 `a–d` 用**图坐标同一 x** 放置（`plabel()`），保证左对齐且左侧无任何文字；网格 `left=0.135`。

---

## 图 8 —— 无人机 RGB（Adak 2023）　→ `pic8\`

- 上游：`knode_maize_rgb_{drought,irrigated}_np_tp1_skip_affine.py`（默认 affine；**当前图读 `*_hyper_skip`**）→ `outputs\kode_result\maize_rgb_{drought,irrigated}\knode_maize_rgb_tp1skip_*_per_trait.csv`
- 数据：`pic8\make_rgb_eval.py → maize_rgb_per_{foldset,timepoint}.csv`（含 Kode + `baseline_rrBLUPt1`（GBLUP 预测 t1 后保持不变）+ `baseline_t1`（实测 t1 保持不变））
- 绘图：`pic8\fig8_maize_rgb_pcc_mse.py → fig8_maize_rgb_pcc_mse.svg`（4 面板：干旱/灌溉 × PCC/MSE；x 轴真实 DAS 间距；灰带标注田间环境事件窗口 DAS 68–75）

---

## 图 9 —— 缺失值（玉米 / 拟南芥）　→ `pic9\`

- 上游：`reproduce\knode_missing_maize_arab_v2.py`（**rank2 + skip + affine**）→ `outputs\reproduce\knode_missing_maize_arab\knode_missing_{maize,arab}_v2_f10.csv`；作者侧 `outputs\reproduce\missing_per_timepoint\`
- 绘图：`pic9\new_fig_missing_compare.py → fig_missing_compare_{maize,arab}.svg`；`pic9\new_fig_missing.py → fig_missing_panel.svg`
- ⚠️ 口径为 **skip**；若要全篇统一到无 skip，需把 `knode_missing_maize_arab_v2.py` 的 `use_skip=True` 改掉后重跑。

---

## 图 10 —— 水稻 in-window（旧版，待定）　→ `pic10\`

- 绘图：`pic10\fig_rice_inwindow.py → fig_rice_inwindow.svg`（两面板：PCC + MSE；Kode 用**无 skip** 的旧 rice 结果，PCC 0.7511）
- ⚠️ 与 `pic2\rice-E2` 版式/口径都不同（后者为 skip、四面板）→ 建议二选一，保留 `rice-E2`。

---

## 表 1 —— 发育算子 A 的遗传力与可预测性　→ `table1\`

- 上游（服务器脚本）：`author_A_stats.py`、`author_A_innerens.py`、`kode_A_sig_trainval.py`
  → 输出 **`outputs\reproduce\author_tp1_perfold\`**（2026-09-15 由 `reproduce\pic-drawing\pic5\author_tp1\perfold\` 迁入）
- 数据/成品：`table1\make_table1_data.py → table1_data.csv`；`table1\make_table1.py → table1.csv`

（表格内容、表注与两条限定见原文档 §表1 ④⑤⑥，数值未变。）

---

## 表 2 —— 图7/图8 三数据集版　→ `table2\`

- 输入：`pic2` 三套 unified 长表（Kode + 作者 tp1）+ `rrBLUP_benchmarks_prediction_accuracies{,_rice}.csv` + **本地** `table2\{maize,arab}\X_all_unrolled_h.csv`（逐性状×时间遗传力，2026-09-15 由 pic-drawing 迁入）
- 数据：`table2\make_table2_data.py → {maize,arab,rice}\{figE8_*_pa_bytp_v4.csv, figE8_*_pa_bytrait_v4.csv, fig5_*_sweep_data.csv, h2_*.csv, rrblup_*.csv}`
- 绘图：`table2\fig7_table2.R <ds>`、`table2\fig8_table2.R <ds>` → `table2\<ds>\Figure7_<ds>.svg / Figure8_<ds>.svg`
- 图注：`table2\<ds>\Figure{7,8}_caption.md`

---

## 重跑顺序

1. （仅当方法结果更新时）跑对应的**上游方法脚本**（Kode / 作者 / 基线）
2. 跑该图的**汇总脚本**（生成图专用数据 CSV）
3. 跑该图的**绘图脚本**（只出 SVG）
4. 跑 §0.1 的**字体复核命令**，确认字体族 / 字重 / 字号没有跑偏

---

## 退役与搬迁记录（2026-09-15）

- **`reproduce\pic-drawing\` 整体退役**：`pic-code\` 已不再引用它。依赖收敛动作：
  1. 拟南芥消融（表 + 绘图脚本 + SVG）→ `pic-code\pic6\arab\`（`pic6\make_component_ablation_from_summary.py` 路径同步）
  2. 逐性状遗传力 `X_all_unrolled_h.csv` → `pic-code\table2\{maize,arab}\`（`make_table2_data.py` 路径同步）
  3. 表1 的作者 per-fold 输入 → `outputs\reproduce\author_tp1_perfold\`（`make_table1_data.py` 路径同步）
  4. 缺失值图两脚本 → 新建 `pic-code\pic9\`（`OUT` 改为本目录）
- **删除的图脚本**（旧管线/重复件）：`fig7_maize.R`、`fig7_arab.R`、`fig8_maize.R`、`fig8_arab.R`、`figE7_maize.R`、`figE7_arabidopsis.R`、`make_fig5_sweep_data.py`（→ 已被 `table2` 取代）；`fig_maize_515_knode_four.R`、`fig_maize_515_knode_mse_box.R`、`new_fig_drone_combined.R`（旧无人机 515 数据集）；以及根目录 `new_fig11_fig12.py`、`new_fig_extrap_all.py`、`new_fig_extrap_maize.py`、`new_fig_extrap_maize2.py`、`new_fig_panels.py`、`plot_extrap_maize_t20fade.py`。
- `reproduce\pic-code\` 根目录现在只保留 `README.md` + `pic2…pic10` + `table1`/`table2`。

---

## 目录一览

```
reproduce\pic-code\
├─ README.md
├─ pic2\  图2（玉米 4 面板）+ arab-E1\（补充图1）+ rice-E2\（补充图2）
├─ pic3\  图3 条件阶梯
├─ pic4\  图4 rank 分析
├─ pic5\  图5 动态基线（玉米/拟南芥/水稻 三张）
├─ pic6\  图6 消融（玉米）+ arab\（拟南芥）
├─ pic7\  图7 外推（玉米）
├─ pic8\  图8 无人机 RGB
├─ pic9\  图9 缺失值（玉米/拟南芥）
├─ pic10\ 图10 水稻 in-window（旧版，待定）
├─ table1\ 表1 算子 A 遗传力/可预测性
└─ table2\ 表2（maize/arab/rice 各 Figure7/Figure8）
```
