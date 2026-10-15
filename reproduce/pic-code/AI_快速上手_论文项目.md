# 论文项目 · 接手 AI 快速上手说明（Kode / 旧称 KNode）

> 你手上有：① 论文 PDF（主稿 + 作者原始论文 dynamicGP）；② 本说明。
> 本说明只讲到"能看懂论文、能改论文"所必需的程度；所有细节都给了**仓库内路径**，需要时按路径查。
> 主要锚点目录：`D:\pythonproject\knode\reproduce\pic-code\`（= 论文所有图/表的数据链路与画图脚本，**这是全项目最规整的一层**）。

---

## 0. 三句话抓住重点

1. **方法名统一用 Kode**（早期文档里叫 KNode，正文请勿混用）。
2. **方法一句话**：只给「基因型（GRM 的 PCA）+ 第一个时间点的表型（TP1）」，用一个超网络生成**该基因型专属的低秩连续时间算子**，再把状态沿真实时间轴传播到全部未来时间点。同一个算子支持两种用法：
   - **recursive**：只用算子自滚（自主预测，主力场景）；
   - **iterative**：每一步都喂入真实状态（teacher-forced，作为**精度上界 / 辅助场景**，不能当自主预测能力宣传）。
3. **主结论**：四个数据集上 Kode 在长程纵向预测上优于作者的 dynamicGP-MegaLMM+TP1 与四种代表性动态基线；有效动力学维度是**低秩饱和**（措辞规矩见 §7）。

---

## 1. 方法核心（读懂论文这几条就够）

| 组元 | 说明 | 代码 |
|---|---|---|
| 条件输入 | `cond = [z_i(GRM-PCA, D_G), y_i(t1)]`，Z-score / PCA 在训练折上拟合 | `reproduce\{maize_magic,arab,rice,maize_RGB}\knode_*_np_tp1_affine.py` |
| 超网络 | `Linear(D_G+P,128)+Tanh → (U_i, V_i, b_i)`，rank = r | 同上 `HyperODEModel` |
| 算子 | `dh/dt = h·VᵀUᵀ + b`（rank-r 低秩 + **affine drift**）；**forward 里没有 t**，是自治连续时间系统 | 同上 |
| 传播 | 用归一化后的**真实时间差**做显式 Euler（`euler_integrate`）；更精确的传播（0.25 天细步 / 矩阵指数闭式解）已验证 `fine ≈ exact` | **服务器** `/data/fandata/dashuaige/reproduce/kode_interp_missing.py`（本地没有，结论见 `outputs/reproduce/kode_interp_missing/`） |
| 训练目标 | 整轨迹 MSE；**验证/测试指标排除 t0** | 主模型脚本 |
| 集成 | inner-5（4/5 训练 + 1/5 早停），测试预测 = 5 折平均 | 主模型脚本 |

**与作者方法的关系**：作者 dynamicGP 用 DMD 学一个**离散单步转移矩阵 A**；Kode 学的是**连续时间低秩算子**（同一族动力学的生成元），因此可以：任意 Δt 传播、跨缺口积分、把"锚点"挪到任意时间点。这一点是论文的机制性差异，也是图 4 / 图 7 的理论依据。

---

## 2. 数据（4 个主数据集 + 2 个备选环境）

| 数据集 | N | T | P | 时间轴 | 在论文里的角色 |
|---|---|---|---|---|---|
| 玉米 MAGIC | 330 RIL | 25 | 50 | DAS 15–47 | 主图 2/3/4/5/6/7 + 表 1 |
| 拟南芥 | 382 | 14 | 45 | DAS 8–21 | 图 2/3/4/5/6/7 + 表 1 |
| 水稻 indica（control） | 226 | 13 | 20 | DAS 30–42 | 图 2/3/4/5/6/E3-E4 |
| 田间玉米 UAV（drought） | 509 RIL | 12 | 19 | DAP 56–99 | 图 3/4/5/6/7/8 |
| 备选：水稻 salt / UAV irrigated | 同上 | 同上 | 同上 | 同上 | 只在需要时的第二环境（未进主图） |

数据文件（每个目录 = 表型 + CV 折 + GRM）：

| 数据集 | 目录 | 关键文件 |
|---|---|---|
| 玉米 MAGIC | `data\maize_magic\` | `maize_trait_BLUP_50_NArmv_traits_afterClustering.csv`、`cross_validation_folds_w_validation_set_maize.csv`、`GRM_330x330_methodA_from552.csv` |
| 拟南芥 | `data\arabidopsis\` | `BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv`、`cross_validation_folds_w_validation_set_arabidopsis.csv`、GRM 382 |
| 水稻 control | `data\rice_salinity\` | `rice_indica_control_traits.csv`、`rice_cv_10fold.csv`、`rice_GRM_226x226.csv` |
| 田间玉米 UAV | `data\maize_RGB\` | `maize_rgb_drought_traits.csv`、`maize_rgb_drought_cv_10fold.csv`、`geno_GRM.csv` |

> 作者（dynamicGP，David Hobby）原论文的对照方法在正文写作里统一记作 **tp1+MegaLMM+dynamicGP**；本地 `data\` 只有这 4 个主数据集（另有 `maize_drought` / `maize_multispectral` / `sesame` 等旁支数据，未进主图）。
> 数据集来源说明见 `数据集来源.md`、`reproduce\rice_dataset_description.md`、`reproduce\maize_uav_rgb_dataset_description.md`。

---

## 3. 评测协议（**改论文时不要动**，所有数字都建立在这上面）

- **十折**：10 个 fold-set；每折 fold 1–5 训练 / fold 6 测试。
- **折内集成**：inner-5（4/5 训练 + 1/5 早停），test 预测 = 5 折平均。
- **归一化**：scaler **只在训练折 folds 1–5** 上拟合（每性状 min/max），套用到验证/测试；验证/测试不参与任何统计量。
- **MSE 可比的前提**：所有方法（Kode、作者、基线）都在**同一个训练折 per-trait min-max 空间**里算 MSE —— 这是论文敢做横向比较的原因，别破坏它。
- **评价**：排除 t0；指标 `PCC / MSE / RMSE / NMSE / NRMSE / R²`，另有纵向 PCC（`r_long`，固定 genotype×trait 沿时间算相关）。
- **统计**：以 **fold-set 为独立重复单元（n=10）**，percentile bootstrap 10 000 次，逐性状比较后做 BH-FDR；脚本 `reproduce\stats_paired_bootstrap_fdr.py`。
- **图注口径**：图 2/7b 的阴影 = 跨性状 SD；图 5/图 7d 的误差线 = fold-set 间 SD。
- **只出 SVG**。

---

## 4. 论文骨架：图表清单（讲什么 / 代码 / 数据 / 现成 SVG）

### 主图

| 图 | 讲什么 | 画图脚本 | 规整数据 | SVG |
|---|---|---|---|---|
| **图1** 方法示意 | Kode 的输入→算子→传播；与 dynamicGP 的差异 | 手工图（`pic1\kode-pic1.pptx`） | — | `pic1\pic1.svg` |
| **图2** 主结果：纵向预测 | Kode(rec/iter) vs DynamicGP-MegaLMM+TP1 / MegaLMM / RR-BLUP，逐时间点 PCC | `pic2\fig2_maize_pcc.py`、`pic2\arab-E1\fig2_arab_pcc.py`、`pic2\rice-E2\fig2_rice_pcc_v2.py` | `pic2\unified_norm_author_vs_kode_per_{timepoint,foldset}.csv`（三套） | 各目录 `fig2_*.svg` |
| **图3** 条件阶梯 | 条件从 Persistence→Shared→GRM→TP1→Full，逐级加信息谁涨 | `pic3\fig_cond_ladder_maize_v2.py`（+`arab\ rice\ rgb\`） | `pic3\fig_cond_ladder_maize_data.csv` | `fig_cond_ladder_*_v2.svg` |
| **图4** 有效动力学维度 | rank 敏感性 + 算子 A 的可预测性 + 谱能量 | `pic4\fig_rank.py`（+`arab\ rice\ rgb\`） | `pic4\fig_rank_data.csv`、`fig_rank_energy.csv` | `fig_rank*.svg` |
| **图5** 动态基线对比 | Kode vs ST-STP / LSTM / AR-Transformer / ODE-RNN / GRU-ODE-Style（PCC + MSE 箱线/曲线） | `pic5\fig_baselines.py`（+`arab\ rice\ uav\`） | `pic5\fig_baselines_data.csv`、`fig_baselines_mse_long.csv` | `fig_baselines*.svg` |
| **图6** 组件 + 超参消融 | rank-2 低秩、affine drift、激活/深度/宽度/维度/种子 | `pic6\fig_ablation_dual_v2.py`（+三数据集） | `pic6\component_ablation.csv`、`hyperparam_ablation.csv` | `Figure_ablation_dual_*_v2.svg` |
| **图7** 时间外推（forecasting） | 三场景：rec_t1（最严）/ rec_tk（t20 锚点）/ iter | `pic7\fig7_extrap.py`（+`arab\ rice\ uav\`） | `pic7\fig7_maize_{b,c,d}.csv`（及其余三套） | `fig7_extrap_*.svg` |
| **图8** 田间无人机 RGB | 干旱条件下的预测精度（PCC/MSE 两面板） | `pic8\fig8_maize_rgb_pcc_mse_v3.py` | `pic8\maize_rgb_per_timepoint_v2.csv`、`_per_foldset_v2.csv` | `fig8_maize_rgb_pcc_mse_v3.svg` |

### 补充图 / 表

| 项 | 讲什么 | 脚本 | 输出 |
|---|---|---|---|
| **E3** | 逐性状预测精度 vs 遗传力（散点）+ 阈值扫描 | `E3-4\E3.R <maize|arab|rice>` | `E3-4\<ds>\FigureE3_<ds>.svg` |
| **E4** | 阈值扫描：不同精度阈值下 Kode 相对 RR-BLUP 的增益 | `E3-4\E4.R <ds>` | `FigureE4_<ds>.svg` |
| **E-table1** | operator-rank 敏感性（四数据集、锁定 D_G 的 rank 扫描 + D_G×rank 网格） | `E-table1\make_E-table1_rank.py`、`make_E-table1_dgxrank*.py` | `E-table1_rank_paste.md` / `.tex` / `E-table1_dgxrank*.csv` |
| **表1** | 算子 A 的遗传力 + **可预测性 r_pred**（Kode vs dynamicGP） | `table1\make_table1_data.py` → `make_table1.py`（在 `reproduce\pic-drawing\pic-code\table1\`） | `table1.csv` |

### 主结果速览（可被正文直接引用，均取自上述 CSV）

**图2 / 图5 总体（recursive，PCC / MSE；`pic5\table_dynamic_methods_overall.csv`）**

| 数据集 | Kode rec | GRU-ODE-Style | ST-STP | ARTrans | LSTM | ODE-RNN |
|---|---|---|---|---|---|---|
| 玉米 MAGIC | **0.5122 / 0.00759** | 0.4789 | 0.4731 | 0.4307 | 0.4068 | 0.3884 |
| 拟南芥 | **0.4860 / 0.00875** | 0.4695 | 0.4686 | 0.4289 | 0.4466 | 0.3737 |
| 水稻 control | **0.7883 / 0.00588** | 0.7856 | 0.7716 | 0.7557 | 0.7696 | 0.7723 |
| UAV drought | — | 0.5102 | 0.5321 | 0.5208 | — | — |

（Kode **iterative** 上界：玉米 0.8909、拟南芥 0.7597、水稻 0.9244、UAV 0.6103 —— 正文必须显式说明 iterative 是上界。）

**图3 条件阶梯（玉米，PCC / R²）**：persistence 0.3645 / −1.195 → shared 0.3660 / 0.746 → **grm 0.4553 / 0.750** → **tp1 0.4754 / 0.807** → **full 0.5127 / 0.814**。
（可写：GRM 与 TP1 各自都有贡献且**互补**，TP1 的边际贡献更大。）

**图4 / E-table1（rank 敏感性，test PCC）**：玉米 D_G=50：r=1 → 0.505，r=2 → 0.512，r≥3 全部落在 0.502–0.507（**极差 < 0.011**）；拟南芥 0.472–0.487；水稻 0.781–0.789；UAV r=1 → 0.403，r=2 → 0.440，r≥3 约 0.420。→ 结论写法见 §7。

**图6 组件消融（linear vs affine drift，玉米）**：recursive PCC 0.5135 vs 0.5125（几乎相同），但 **R² 中位 −0.068 → +0.039、MSE 0.00946 → 0.00758、NRMSE 0.119 → 0.0837、r_long 0.817 → 0.871**。
（可写：affine drift 的收益主要体现在**绝对误差与长程相关性**，而不在 PCC。）

**图7 外推（玉米；PCC / MSE）**

| 方法 | iter | rec_tk（t20 锚点） | rec_t1（最严） |
|---|---|---|---|
| Kode | **0.9203 / 0.00375** | **0.8276 / 0.01320** | **0.3968 / 0.02340** |
| dynamicGP-MegaLMM+TP1 | 0.5087 / 0.05095 | 0.4934 / 0.05207 | 0.3325 / 0.05812 |

**表1（算子 A）**：玉米 Kode `h²_train 0.863 / r_pred 0.713` vs dynamicGP `0.324 / 0.458`；拟南芥 Kode `0.540 / 0.669` vs `0.163 / 0.435`。⚠️ 表注必须保留（见 §7）。

---

## 5. `pic-code` 目录导航（**三层数据链路**）

```
reproduce\pic-code\
├─ README.md                 ← 每张图/表的完整数据链路总索引（需要改图先查它）
├─ pic1\      方法示意图（pptx + svg）
├─ pic2\      图2 纵向预测（+ arab-E1\ rice-E2\）
├─ pic3\      图3 条件阶梯（+ arab\ rice\ rgb\）
├─ pic4\      图4 rank（+ arab\ rice\ rgb\）
├─ pic5\      图5 动态基线（+ arab\ rice\ uav\）
├─ pic6\      图6 消融（+ arab\ rice\ uav\）
├─ pic7\      图7 外推（+ arab\ rice\ uav\）
├─ pic8\      图8 无人机 RGB
├─ E3-4\      补充图 E3/E4（+ maize\ arab\ rice\）
├─ E-table1\  拓展表（CSV + 排版 md/tex）
├─ result-pic\ 各图画图脚本直接读的数据文件（= 第④层，**改图先来这里**）
└─ related\_document\  ＊原始脚本归档（只存证，**不能直接跑**；索引见 README §1B）
```

**每个图目录里的两类脚本**（记住这个约定，其余都好办）：

| 类型 | 命名 | 干什么 | 读什么 | 写什么 |
|---|---|---|---|---|
| ③ 规整脚本 | `make_*.py`（或 `unified_norm_eval*.py`） | 把 `outputs\` 里的原始结果整理成画图用的表 | `outputs\...` | 图目录里的 `*.csv` |
| ⑤ 画图脚本 | `fig*.py` / `fig*.R` / `E3.R`、`E4.R` | 读规整表 → 出 SVG | 规整 `*.csv` | `*.svg` |

---

## 6. 重跑规则（改图/改数时**必须**按这个判断）

| 你改了什么 | 需要重跑 |
|---|---|
| 字体 / 配色 / 轴范围 / 图例 / 标注 | **只跑 ⑤ 画图脚本** |
| 聚合口径（SD 定义、纳入哪些性状、时间点） | ③ + ⑤ |
| 换模型 / 换折 / 换数据 / 换超参 | ① 原始脚本（唯一需要训练的一步）→ ③ → ⑤ |

**红线**：
1. **不要动 `outputs\` 里的原始结果文件**（那些是 ① 层的产出，重跑代价高、且是论文数字的唯一来源）。
2. **不要自创输出格式**，不要删模板原有产物，不要把"逐性状/逐时点长表"简化成"只有汇总表"（项目曾因此白跑一次全量实验）。
3. 图表只出 **SVG**；需要 PNG 预览可以临时生成但不要作为交付。
4. 训练/重跑一律在服务器跑（`fan@202.119.209.206`，工作区 `/data/fandata/dashuaige/`），不要在本机跑十折。

---

## 7. 写论文的表述红线（**最容易被审稿人抓的几条**）

1. **rank 结论写"低秩饱和"（low-rank saturation），不要写"最优 rank = 2"**。
   证据：玉米 r=1 → 0.505、r=2 → 0.512、r=3/4/8/20/50 → 0.502–0.507（极差 < 0.011），是**平台不是尖峰**；同理适用于 GRM PCA dim、hidden、depth、激活函数 —— 一律写"稳健平台 + 取区间中部代表值"，给出区间内差异量级。
2. **MSE 横向比较的三个前提**必须同时成立：同一训练折 min-max 空间、同一 fold 划分、同一时间点。作者方法复现时 DMD 必须 `scale="none"`（原作者默认的 `min.max` 是 per-subset，会泄漏验证集表型范围）。
3. **表 1 的"算子 h²"不可与 dynamicGP 的 validation h² 直接比较**（Kode 的 A 是模型从 `[GRM, y1]` 生成的，其 h² 衡量的是"生成算子与 GRM 的一致性"）；该表以 `r_pred`（算子可预测性）为主指标。**这条脚注必须保留。**
4. **iterative ≠ 自主预测**：iterative 每步喂真值，只能作为**上界 / observation-assisted** 场景；正文禁止用它做"我们的模型能自主预测多远"的论证（用 recursive）。
5. **少性状/单性状**：三个数据集一致"整体不划算"（玉米 68%、水稻 65%、拟南芥 42% 的性状变差；平均 PCC −0.061/−0.032/−0.017），例外集中在**面积/大小类**性状。正确表述是"**任意子集输入都能出结果、输入性状越多越稳定**"，不能写成"性状越多一定越好"或"单性状更好"。
6. **dynamicGP「20 点比 25 点好」不能用来说明"训练数据多了变差"**：那是**低秩 DMD 在更长发育窗口上的全局折中**（已有 oracle-DMD 诊断支撑）；写成机制解释，别写成反直觉卖点。
7. **泛化边界**要写清：Kode 的算子由 `[GRM, TP1]` 生成，所以 (a) 需要 TP1（只有 SNP 的个体要走"预测算子"支线）；(b) `y1` 来自结构化的 HTP 表型，不能外推到任意作物/单位/测量体系；(c) 目前没有置信区间/概率输出的校准验证。
8. **时间点的边界**：主模型把 TP1 固定为锚点；"任意时间点输入 / 任意日期查询"属于**连续时间算子的自然能力**（细步长与矩阵指数已验证 `fine ≈ exact`），但**训练分布只在数据网格日上有监督**，正文若要写这条能力，必须同时写明"非网格日期未验证"。

---

## 8. 项目里还有哪些文档值得看（连同论文一起交接）

| 文件 | 内容 |
|---|---|
| `reproduce\pic-code\README.md` | **图/表数据链路总索引**（改图必查，253 行） |
| `reproduce\pic-code\result-pic\README.md` | 每张图 → 画图脚本 → 读哪个 CSV 的对照表 |
| `HANDOVER.md`（根目录 / `reproduce\`） | 项目级交接：历史、服务器、环境、踩坑 |
| `结果.md` | 早期 dynamicGP 复现结果汇总（数字口径说明） |
| `创新点.md` | 方法卖点与与作者工作的差异梳理 |
| `数据集来源.md` | 四个数据集的出处与预处理 |
| `reproduce\TP1only_实验报告.md`、`reproduce\STSTP_单性状实验报告.md` | 两组"少输入"实验的完整报告（§7-5 的依据） |
| `AGENTS.md`（根目录 / `reproduce\`） | 本项目的硬性工作规则（输出三层、SVG、模板照抄、启动前查重…） |

---

## 9. 建议的工作流（接手后怎么干）

1. 先读论文 + 本说明，产出一张映射表：**每一条结论句 ↔ 哪张图/表 ↔ 哪个 CSV**（`result-pic\README.md` 能一张表填完）。
2. 若只是**文字 / 图注 / 排版**：不要碰代码，直接改稿；涉及数字时从 §4 的 CSV 里读，**不要凭记忆**。
3. 若要**换图/换数字**：先判断改动落在 §6 的哪一层，再决定重跑范围；训练类重跑一律上服务器。
4. 每次改完图，同步更新 `pic-code\README.md` 对应那一节的"⑤ 画图脚本 / ⑥ 输出"。
5. 有疑问先问清楚再动手：**改口径/删文件/重跑 = 必须先确认**。
