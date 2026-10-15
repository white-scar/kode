# Maize UAV RGB (Adak et al. 2023) — KNode 十折结果说明

## 1. 数据
- 来源：Dryad `doi:10.5061/dryad.1vhhmgqwm`（Adak et al. 2023, J Exp Bot 74:5307–5326）
- 表型：`YYCD18_RGB.csv`( 干旱 ) / `YYCI18_RGB.csv`( 灌溉 )，每次飞行每小区 32~35 个 VI
- 基因型：复用 `data/maize_drought/SNP/geno.numerical.csv`（11,333 个 AgR 标记）与 `geno_GRM.csv`
- 交集：干旱 N=522、灌溉 N=520
- 时间点：**12 个** = DAP 42, 56, 61, 64, 68, 71, 75, 78, 82, 84, 90, 99
  （去掉 DAP 27/34 高缺失点、110/127 低 h2 点）

## 2. 性状选择（按作者 Arabidopsis_data_prep.R 的规则）
1. 性状两两做 Mantel 相关（基因型间距离矩阵）
2. >= 0.97 建邻接矩阵 -> Louvain 聚类
3. 每簇取平均遗传力 h2 最高的性状
4. 干旱(18 个) 与 灌溉(20 个) 取**交集 -> 17 个性状**:
   BCC, BRVI, COM1, EBI, ExG, ExGR, G-B, G-R, GCC, MExG, MRCC, NRMBI, R-B, R/B, RCC, RGBVI, TGI
- 这 17 个在 12 个点上 h2 均值约 0.25（远高于多光谱 VI 的 0.08）

## 3. 协议（用户确认口径）
- 10 个 fold-set，每个 6 折：fold1-5 训练 / fold6 测试（5:1）
- 训练集内部 5 折循环（4/5 训练 + 1/5 早停），测试预测 = 5 个 inner 模型平均
- 归一化：**只用训练折 per-trait min-max**，套用到训练/早停/测试
- 指标：PCC 与 MSE，MSE 在训练折 min-max 归一化空间
- baseline：rrBLUP(GBLUP) 只预测第一个点，之后保持该预测值（persistence）

## 4. 结果（10 折均值，排除输入点 tp1；除注明外）
| env | method | PCC_mean | MSE_mean | PCC_pooled |
|---|---|---|---|---|
| drought | ode_hyper_skip (rec) | 0.4685 | 0.014182 | 0.7534 |
| drought | iter_hyper_skip (iter) | 0.6258 | 0.014652 | 0.7428 |
| drought | baseline | 0.3454 | 0.056077 | 0.2165 |
| irrigated | ode_hyper_skip (rec) | 0.3708 | 0.014366 | 0.7602 |
| irrigated | iter_hyper_skip (iter) | 0.5225 | 0.012361 | 0.8005 |
| irrigated | baseline | 0.1167 | 0.058570 | 0.2828 |

### 两种 overall 口径（都要在论文里说明）
- `PCC_mean` = 逐 (trait x 时间点) 算 PCC/MSE 再平均 —— **推荐主口径**，与作者/以往脚本一致
- `PCC_pooled` = 把所有 trait x 时间点的值拼起来算一个相关系数 —— 会被"性状间水平差异"抬高，偏高
- 文件：`maize_rgb_overall_full10.csv`（pooled）/ `maize_rgb_overall_full10_meanconv.csv`（mean）

## 5. DAP 68/71/75 的田间异常（必须如实说明）
### 现象
- DAP 68(5/22)、71(5/25)、75(5/29) 三个飞行点上，多个 VI 出现"先整体抬升、再整体塌陷"的反常波动：
  - TGI: 64->68 +50.2, 68->71 +40.2, 71->75 **-114.3**（其它点典型变化 ±3）
  - MExG: +9.1 / +7.6 / **-24.4**（其它点 ±1~2）
  - BCC: +0.089 / +0.009 / -0.060（其它点 ±0.003）
- **所有基因型同步平移**（Ki3/LAMA_YC/LH195/LH82/NC356 的 BCC 在 DAP68 全部 +0.083~+0.091），无基因型特异性

### 关键证据：不是数据错误，而是全田环境事件
1. **干旱与灌溉两组在同一批日期同步出现**（如 TGI 71->75: 干旱 -114.3 / 灌溉 -85.6）
2. **有辐射定标的多光谱传感器同期也测到**（NDVI 5/19->5/21: 0.248 -> 0.348）
3. 与 Adak et al. (2023) 的核心叙事吻合：热胁迫峰值期的**时序反射表型**揭示了热相关基因 hsp18f，
   而该时间窗正是 5 月下旬
4. "先凸后塌"符合短期高温 -> 萎蔫 -> 恢复的进程

### 为什么它会压低指标
该波动是**全田共享**的（非基因型特异），KNode/dynamicGP 只能从基因型 + tp1 预测基因型特异轨迹，
无法预测这种整片田同步的环境波动，因此在逐点上表现为误差尖峰：
- iter 在 DAS 68 MSE=0.0781（正常点 0.005~0.02）
- baseline 在 DAS 68/71 MSE=0.17/0.22

### 论文写法建议
保留全部 12 个点，但明确写清：
> 在 DAP 68–75（5/22–5/29）期间，所有基因型与两个水分处理同步出现反射率整体偏移，且这一偏移在
> 有辐射定标的多光谱数据中同样存在；结合该时段与已报道的热胁迫峰值期吻合（Adak et al., 2023），
> 我们将其归因于该时段的全田环境事件（高温/萎蔫）导致的系统性表型偏移，而非基因型效应或数据错误。
> 由于该偏移为全田共享、无法由基因型信息预测，它在所有方法中同步表现为逐点误差升高；我们将其
> 完整纳入统计而非作为离群点剔除，以如实反映真实田间条件下的模型表现。

### 消融建议（可选）
另跑一版**去掉 DAP 68/71/75（保留 9 点）**的结果作为补充：去掉后 iter 的 MSE 从 0.01465 降到 0.0083
（rec 0.01417 -> 0.01371），能说明"该波动确实是全田共享的环境项、而非模型的基因型预测失败"。

## 6. 文件
- 脚本：`reproduce/knode_maize_rgb_drought_np_tp1_affine.py`
- 数据：`data/maize_rgb/`（traits + cv_10fold + geno_GRM）
- 结果：`outputs/kode_result/maize_rgb/`
  - `maize_rgb_by_time_full10.csv`（12 点 x 3 方法）
  - `maize_rgb_by_trait_full10.csv`（17 性状 x 3 方法）
  - `maize_rgb_overall_full10.csv` / `maize_rgb_overall_full10_meanconv.csv`
  - `run_full10.log`
- 性状筛选：`data/maize_multispectral/maize_rgb_selected_traits_{drought_rgb,irrigated_rgb}.csv`
