# ST-STP 单性状全覆盖实验报告（玉米 MAGIC / 水稻 indica / 拟南芥）

> 日期：2026-10-11　|　计划书：`C:\Users\27342\Downloads\ststp计划书.md`

**一句话结论**：把性状拆开单独建模，整体上不划算（三个数据集一致）；但约三成性状单性状更好，所以产品形态应该是“任意子集输入都能输出、输入越多越稳定”。

---

## 一、为什么做这个实验

上一轮 **TP1-only 实验（E1 all / E2 single / E3 few）** 已经比较过“单性状专属模型”和“全性状联合模型”，但那次只有 **少数挑选过的性状**（每个数据集 1–5 个）参与单性状对比，而且部分结论建立在 **5 个模型族的平均** 上，存在**选择偏差**：被挑中的往往是先前表现好的性状。

因此本轮做一次**全覆盖、同模型、同口径**的验证：

- 同一份数据、同一套 CV（folds1–5 训练 / fold6 测试）、同一 `iters=0,1,2`、同一 inner-5 集成、同一 `seed=42`；
- **只用 `ststp` 一个模型族**，只比较 **ST-STP 单性状 vs ST-STP 全性状**，不再混入 5 族平均；
- 每个性状**单独训练一个 P=1 模型**，并把全性状模型的**同一个性状那一列**作为配对基线；
- 覆盖全部性状：玉米 50、水稻 20、拟南芥 45，共 **115 个性状**，不挑不选。

要回答的问题只有一个：

> **在同一个性状上，“只吃自己 TP1 的专属模型”能不能打败“吃下全部性状的联合模型”？如果能，产品端就可以只要求用户提供少量指标。**

---

## 二、实验怎么设计

### 2.1 两组对照（唯一变量 = 输入性状数）

| 组 | 输入 | 输出 | 来源 |
|---|---|---|---|
| **S 组 single** | 该性状自身 TP1 的归一化值，`P=1`、`COND_IN=1` | 该性状未来全部时点 | 本轮新训 115 个 P=1 模型 |
| **A 组 all（基线）** | 全部 P 个性状的 TP1 | 全部 P 个性状的未来轨迹 | 上一轮 `knode_tp1only_<ds>/all/` 的 **ststp** 分量 |

**公平比较规则（关键）**：两组只在**同一个目标性状**上比——把 A 组结果里 `trait=该性状` 的那一列取出来，与 S 组该性状的结果逐时点配对，保证**同折、同测试材料、同时点、同归一化空间**。

### 2.2 三个数据集

| 数据集 | `dataset` | N（分析用） | 时点 T | 性状 P | 评价时点（排除 t0） | 时间轴 |
|---|---|---|---|---|---|---|
| 玉米 MAGIC | `maize_magic` | 330 | 25 | **50** | **24** | DAS 15–47 |
| 水稻 indica（control） | `rice_indica_control` | 226 | 13 | **20** | **12** | DAS 30–42 |
| 拟南芥 | `arabidopsis` | 382 | 14 | **45** | **13** | DAS 8–21 |

### 2.3 统一协议

- **只跑 ststp**：不跑 LSTM / ARTransformer / ODE-RNN / GRU-ODE-Bayes；
- **完全不用 GRM/SNP**：条件向量就是 TP1 本身（单性状时 `COND_IN=1`）；
- **CV**：10 个 fold-set，本轮取 `iters=0,1,2` 三个 outer iteration；每折 folds1–5 训练 / fold6 测试；
- **折内集成**：inner-5（4/5 训练 + 1/5 早停），测试预测 = 5 个 inner 模型平均；
- **归一化**：只用**训练折**的 per-trait min-max，套用到验证/测试；测试标签不参与任何统计量；
- **训练**：epochs=1000，Adam(lr 1e-3, wd 1e-4)，梯度裁剪 1.0，固定 `seed=42`；
- **评价**：排除 t0，只评价未来 24 / 12 / 13 个真实观测时点；
- **指标**：每个 `fam × 时点 × 性状` 都算 `pcc, mse, RMSE, R2, NMSE, NRMSE, y_true, y_pred, SD`，无法计算的情形保留行并写 `NaN + nan_reason`。

### 2.4 判定规则

`ΔPCC = PCC_single − PCC_all`，`ΔMSE = MSE_single − MSE_all`（**MSE 越低越好**）。预先固定 `|ΔPCC| < 0.01` 记为**近似持平**，仅作描述性统计；3 个 iteration 不足以声称统计显著性，本轮不做显著性检验。

### 2.5 产物（每个性状 7 件；每个数据集另加 3 件对照）

| 文件 | 内容 |
|---|---|
| `metrics_by_trait_time.csv` | 主表：`fam,time,trait,pcc,mse,RMSE,R2,NMSE,NRMSE,y_true,y_pred,SD,nan_reason` |
| `metrics_by_trait_time_per_iter.csv` | 同上，多一列 `iteration`（逐 iteration 明细） |
| `metrics_by_trait_time_std.csv` | 3 个 iteration 间各指标的 SD |
| `metrics_overall.csv` | 该性状在全部未来时点 × iteration 上的总体指标（逐单元等权平均） |
| `predictions_long.csv.gz` | 逐材料预测：`iteration,fam,bio_id,time,trait,y_true,y_pred` |
| `run_config.json` | 数据集/性状/iter/epochs/seed/归一化/指标定义/test_ids_by_iter/起止时间 |
| `train.log`（+ `_run.log`） | 运行日志 |

三件对照产物（`comparison_*`）：逐性状汇总（50 / 20 / 45 行）、逐性状逐时点明细（1200 / 240 / 585 行）、`comparison_summary.md`。

---

## 三、代码位置

### 3.1 主脚本（三份同构：从上一轮 TP1-only 脚本逐行复刻，只改数据集、性状清单、输出目录、脚本名）

| 数据集 | 本地路径 | 服务器路径 |
|---|---|---|
| 玉米 MAGIC | `D:\pythonproject\knode\reproduce\maize_magic\knode_ststp_single_maize.py` | `/data/fandata/dashuaige/reproduce/knode_ststp_single_maize.py` |
| 水稻 indica（control） | `D:\pythonproject\knode\reproduce\rice\knode_ststp_single_rice.py` | `/data/fandata/dashuaige/reproduce/knode_ststp_single_rice.py` |
| 拟南芥 | `D:\pythonproject\knode\reproduce\arab\knode_ststp_single_arab.py` | `/data/fandata/dashuaige/reproduce/knode_ststp_single_arab.py` |

### 3.2 驱动与对照脚本

| 作用 | 本地路径 | 服务器路径 |
|---|---|---|
| 批量驱动（逐性状串行/并行、断点续跑、冒烟隔离） | `D:\pythonproject\knode\reproduce\_ststp_single_batch.py` | `/data/fandata/dashuaige/reproduce/_ststp_single_batch.py` |
| 单性状 vs 全性状对照（生成 3 件 `comparison_*`） | `D:\pythonproject\knode\reproduce\_ststp_single_compare.py` | `/data/fandata/dashuaige/reproduce/_ststp_single_compare.py` |

启动命令（以水稻为例）：

```bash
cd /data/fandata/dashuaige/reproduce
/data/fandata/WWC/anaconda3/envs/dashuaige/bin/python _ststp_single_batch.py --dataset rice --parallel 4
/data/fandata/WWC/anaconda3/envs/dashuaige/bin/python _ststp_single_compare.py --dataset rice
```

---

## 四、结果

### 4.1 总览：逐性状配对比较（唯一公平口径）

| 数据集 | 性状数 | 提升 (ΔPCC>+0.01) | 近似持平 | 下降 (ΔPCC<−0.01) | ΔPCC 均值 | ΔPCC 中位 | ΔPCC 范围 | ΔMSE 均值 | MSE 变好的性状 |
|---|---|---|---|---|---|---|---|---|---|
| 玉米 MAGIC | 50 | 14（28%） | 2（4%） | **34（68%）** | **-0.0611** | -0.0462 | -0.3941 ~ +0.2018 | +0.00014 | 19 |
| 水稻 indica（control） | 20 | 7（35%） | 0（0%） | **13（65%）** | **-0.0318** | -0.0402 | -0.1902 ~ +0.0992 | +0.00078 | 7 |
| 拟南芥 | 45 | 16（36%） | 10（22%） | **19（42%）** | **-0.0173** | -0.0054 | -0.3592 ~ +0.1930 | +0.00006 | 25 |

> 单性状模型每个数据集都平均更差：PCC 分别低 0.0611 / 0.0318 / 0.0173，MSE 分别高 0.00014 / 0.00078 / 0.00006。

### 4.2 整体口径（与 `metrics_overall.csv` 同定义：逐 性状 × 时点 × iteration 单元等权平均）

| 数据集 | 组 | PCC | MSE | RMSE | R² | NMSE | NRMSE |
|---|---|---|---|---|---|---|---|
| 玉米 MAGIC | 单性状（50 个模型平均） | 0.3403 | 0.00695 | 0.07554 | 0.1605 | 0.8395 | 0.19542 |
| 玉米 MAGIC | **全性状联合 ststp** | **0.4014** | **0.00681** | **0.07573** | **0.1486** | **0.8514** | **0.19525** |
| 水稻 indica（control） | 单性状（20 个模型平均） | 0.7397 | 0.00636 | 0.07316 | 0.5275 | 0.4725 | 0.20974 |
| 水稻 indica（control） | **全性状联合 ststp** | **0.7715** | **0.00558** | **0.07037** | **0.5732** | **0.4268** | **0.20420** |
| 拟南芥 | 单性状（45 个模型平均） | 0.4129 | 0.00748 | 0.07883 | 0.2141 | 0.7859 | 0.18788 |
| 拟南芥 | **全性状联合 ststp** | **0.4302** | **0.00742** | **0.07986** | **0.2077** | **0.7923** | **0.18938** |


> ⚠️ **口径提醒**：玉米与拟南芥的 R² / NMSE 出现了“反向”（单性状略高），这不是矛盾——R² 是**逐性状相对其自身方差**归一化后再平均，单性状模型在低方差、难预测的性状上相对损失更小；而 PCC 与 MSE 在**同一绝对尺度**上计算，三个数据集一致指向全性状联合更好。因此正文以 **PCC + MSE 为主指标**，R² / NMSE / NRMSE 作补充并注明口径。

### 4.3 ΔPCC 随时间点的走向（所有性状平均）

| 数据集 | 首个评价时点 | 前 1/4 时点均值 | 后 1/4 时点均值 | 最后一个时点 | 最差时点 | 末段 1/4 窗口改善的性状 |
|---|---|---|---|---|---|---|
| 玉米 MAGIC | +0.1356 | +0.0637 | **-0.1121** | -0.0893 | -0.1333 | 9/50 |
| 水稻 indica（control） | +0.0436 | +0.0138 | **-0.0710** | -0.0778 | -0.0778 | 6/20 |
| 拟南芥 | +0.0606 | +0.0273 | **-0.0701** | -0.0924 | -0.0960 | 13/45 |

**共同的形状**：**早期单性状更好、中期翻转、越往后越差**。最典型的是玉米：第 1–2 个评价时点 ΔPCC 达 +0.136 / +0.141，从第 9 个时点起稳定在 −0.09 ~ −0.13。说明单性状模型在“只推一两步”时够用，一旦要长期递推，缺少其它性状的上下文就开始吃亏。

### 4.4 代表性性状

**玉米 MAGIC —— 提升最多 5 个**

| trait | PCC single | PCC all | ΔPCC | ΔMSE |
|---|---|---|---|---|
| `side.geometry.vis.hull.points` | 0.4565 | 0.2548 | **+0.2018** | -0.00186 |
| `top.intensity.fluo.hsv.v.skewness` | 0.2126 | 0.1267 | **+0.0859** | -0.00034 |
| `side.geometry.vis.leaf.curvature.avg` | 0.2753 | 0.2083 | **+0.0669** | -0.00013 |
| `top.geometry.fluo.hull.points` | 0.1991 | 0.1465 | **+0.0526** | +0.00006 |
| `top.intensity.vis.hsv.h.histogram.s_avg.bin.07.076_089` | 0.3479 | 0.2956 | **+0.0523** | -0.00007 |

**玉米 MAGIC —— 下降最多 5 个**

| trait | PCC single | PCC all | ΔPCC | ΔMSE |
|---|---|---|---|---|
| `top.intensity.vis.hsv.normalized.v.histogram.bin.03.025_038` | -0.0262 | 0.3679 | **-0.3941** | +0.00011 |
| `top.intensity.vis.hsv.normalized.v.histogram.bin.07.076_089` | 0.1282 | 0.4132 | **-0.2850** | +0.00036 |
| `top.intensity.vis.hsv.h.skewness` | 0.2502 | 0.5117 | **-0.2615** | +0.00196 |
| `top.intensity.vis.hsv.v.histogram.s_avg.bin.07.076_089` | 0.1447 | 0.3887 | **-0.2440** | +0.00120 |
| `top.texture.fluo.hsv.s.mean` | 0.0664 | 0.2857 | **-0.2193** | +0.00014 |

**水稻 indica（control） —— 提升最多 5 个**

| trait | PCC single | PCC all | ΔPCC | ΔMSE |
|---|---|---|---|---|
| `TV_Area` | 0.9342 | 0.8350 | **+0.0992** | -0.00163 |
| `PSA` | 0.9657 | 0.9061 | **+0.0597** | -0.00093 |
| `SV2_Center_Y` | 0.8353 | 0.7971 | **+0.0381** | -0.00149 |
| `TV_Boundary_Ratio` | 0.8812 | 0.8447 | **+0.0365** | -0.00067 |
| `SV2_Area` | 0.9262 | 0.9014 | **+0.0248** | -0.00026 |

**水稻 indica（control） —— 下降最多 5 个**

| trait | PCC single | PCC all | ΔPCC | ΔMSE |
|---|---|---|---|---|
| `SV2_Compactness` | 0.5591 | 0.7493 | **-0.1902** | +0.00344 |
| `SV1_Compactness` | 0.5391 | 0.6642 | **-0.1250** | +0.00429 |
| `SV1_Caliper_Length` | 0.5898 | 0.6868 | **-0.0970** | +0.00194 |
| `SV1_Convex_Hull_Area` | 0.6924 | 0.7733 | **-0.0809** | +0.00160 |
| `SV2_Convex_Hull_Area` | 0.7450 | 0.8230 | **-0.0780** | +0.00175 |

**拟南芥 —— 提升最多 5 个**

| trait | PCC single | PCC all | ΔPCC | ΔMSE |
|---|---|---|---|---|
| `hsv.s.mean..color.related.trait.based.on.visible.light.top.view.` | 0.5128 | 0.3197 | **+0.1930** | -0.00087 |
| `hull.points..geometry.trait.based.on.fluorescence.top.view.` | 0.3182 | 0.1867 | **+0.1315** | -0.00015 |
| `leaf.width.mean..geometry.trait.based.on.fluorescence.top.view...px.` | 0.2805 | 0.1724 | **+0.1080** | -0.00028 |
| `hull.fillgrade..geometry.trait.based.on.visible.light.top.view.....` | 0.3912 | 0.3324 | **+0.0588** | -0.00031 |
| `cog.x..geometry.trait.based.on.visible.light.top.view...px.` | 0.6348 | 0.5800 | **+0.0548** | -0.00089 |

**拟南芥 —— 下降最多 5 个**

| trait | PCC single | PCC all | ΔPCC | ΔMSE |
|---|---|---|---|---|
| `hsv.h.mean..color.related.trait.based.on.visible.light.top.view.` | 0.2334 | 0.5926 | **-0.3592** | +0.00485 |
| `daily.log.change.rate.of.area..geometry.trait.based.on.fluorescence.top.view...relative.day.` | 0.0382 | 0.2025 | **-0.1643** | +0.00004 |
| `relative.daily.change.of.area..geometry.trait.based.on.fluorescence.top.view...relative.day.` | 0.0582 | 0.2112 | **-0.1530** | +0.00012 |
| `leaf.count..geometry.trait.based.on.fluorescence.top.view.` | 0.0114 | 0.1425 | **-0.1311** | +0.00008 |
| `lab.b.skewness..color.related.trait.based.on.fluorescence.top.view.` | 0.2960 | 0.4138 | **-0.1178** | +0.00128 |

**规律**：单性状赢的多是**面积/大小类**、自身信号强、PCC 本来就高的性状（水稻 `TV_Area` 0.934、`PSA` 0.966）；输得最惨的是**紧实度、卡尺长/边界比、形状与颜色纹理偏度、相对日变化率**这类需要上下文（其它性状、其它时点）才有意义的性状。而且**下行风险远大于上行收益**：赢最多 +0.2018（玉米），输可以到 -0.3941（玉米）/ -0.3592（拟南芥）/ -0.1902（水稻）。

### 4.5 结论（一页版）

1. **大趋势：输入性状越多越稳。** 三个数据集一致（玉米 68%、水稻 65%、拟南芥 42% 的性状在单性状下变差），PCC 与 MSE 指向同一方向；
2. **约三成性状是例外**（玉米 28%、水稻 35%、拟南芥 36% 提升），集中在面积/大小类、自身可预测性高的性状；
3. **风险不对称**：单性状赢的幅度小（≤ +0.20），输的幅度大（可到 −0.36 ~ −0.39），所以**默认应保持全性状联合模型**；
4. **时间上：短期单性状够用、长期必须靠上下文**（ΔPCC 由早期正向转为末段 −0.07 ~ −0.11）；
5. **产品形态建议**：不做“必须交全性状”的硬门槛，而是**开放任意子集输入**——用户给几个性状就出这几个性状的结果；默认推荐全量输入以获得最稳定的长期外推；面积/大小类性状允许单独提供并单独建头。

---

## 五、结果所在位置

### 5.1 服务器（正式结果）

```
/data/fandata/dashuaige/outputs/reproduce/knode_ststp_single_maize/    # 50 个 single_pXX_*
/data/fandata/dashuaige/outputs/reproduce/knode_ststp_single_rice/     # 20 个 single_pXX_*
/data/fandata/dashuaige/outputs/reproduce/knode_ststp_single_arab/     # 45 个 single_pXX_*
```

每个数据集目录下：`single_pXX_<trait>/`（各 7 件产物 + `_run.log`）、`_smoke/`（冒烟，隔离存放）、`batch.log`、`batch_stdout.log`、`compare_stdout.log`，以及三件对照：`comparison_single_vs_all_ststp_by_trait.csv`、`comparison_single_vs_all_ststp_by_trait_time.csv`、`comparison_summary.md`。

全性状基线（对照组）：`/data/fandata/dashuaige/outputs/reproduce/knode_tp1only_{maize,rice,arab}/all/`

### 5.2 本地（同一份拷贝，已下载）

```
D:\pythonproject\knode\outputs\reproduce\knode_ststp_single_maize\
D:\pythonproject\knode\outputs\reproduce\knode_ststp_single_rice\
D:\pythonproject\knode\outputs\reproduce\knode_ststp_single_arab\
```

### 5.3 关键运行事实（便于复核）

| 数据集 | 性状完成度 | 用时 | 并行度 | 产物 |
|---|---|---|---|---|
| 玉米 MAGIC | 50/50 | 4.2 min | 4 | ✅ 全齐 |
| 水稻 indica | 20/20 | 1.7 min | 4 | ✅ 全齐 |
| 拟南芥 | 45/45 | 3.9 min | 4 | ✅ 全齐 |

- 每个性状任务：3 个 outer iteration × inner-5 集成，`device=cuda`，单个性状约 19–20 s；
- 运行日志逐条 `ok`，**无 Traceback、无缺件**，列名与上一轮 TP1-only 逐字一致；
- **旧脚本、旧结果（`knode_tp1only_*`、`knode_dynamic_baselines_full*`）一个字节都没改**；
- 本轮只重跑过一次对照脚本（修 `comparison_summary.md` 里写死的数据集名），**未重训任何模型**，数值未变。
