# 玉米无人机 RGB（UAV RGB, Adak et al. 2023）数据集说明

> 用于 Kode/dynamicGP 复现的玉米 UAV RGB 表型-基因型数据。数字均由 `data/maize_RGB/`、`data/maize_multispectral/`、`data/maize_drought/SNP/` 与原文 PDF（`data/maize_RGB/erad216.pdf`）核对得到。

## 1. 群体与育种

数据来自 **Adak et al. (2023, *Journal of Experimental Botany* 74:5307–5326)**《Phenomic data-driven biological prediction of maize through field-based high-throughput phenotyping integration with genomic data》，数据存放于 Dryad（`doi:10.5061/dryad.1vhhmgqwm`）。

分析群体为 **520 个玉米重组自交系（RIL）**，来自三个双亲杂交组合：

| 组合 | RIL 数 | 亲本特点 |
|---|---|---|
| Ki3 × NC356 | 238 | Ki3 热带，NC356 热带背景/温带选择 |
| LH82 × LAMA | 178 | LH82 温带适应；LAMA 与 Tx740 约 50% 相关 |
| Tx740 × NC356 | 104 | Tx740 热带背景/温带选择 |

2014–2018 年间经**单粒传法（SSD）**在 College Station（夏季繁育圃）与 Weslaco（冬季繁育圃，德州）繁育至 **F7:8**。

## 2. 田间试验

2018 年在**德州 Texas A&M（College Station）同一试验站、同一年份**设置两种水分管理，构成两个环境：**灌溉（irrigated，YYCI）**与**干旱（drought，YYCD）**。

- **播种**：2018-03-14（两个试验同日播种）
- **设计**：随机完全区组设计（RCBD），**2 个重复**；小区为**单行**、行长 **3.81 m**、区间距 **0.76 m**
- 除影像性状外还人工调查了开花期（DTA/DTS）与株高（PHT）

## 3. 无人机影像与正射拼接

| 平台 | 载荷 | 高度 / 分辨率 | 架次 |
|---|---|---|---|
| **RGB**（本文使用） | DJI Phantom 3 Professional，12 MP DJI FC300X | 25 m，≈**1 cm/px** | **16 次**（原始 19 个正射影像，前 3 个为裸地未用） |
| 多光谱（对照） | Tuffwing 固定翼 + MicaSense RedEdge-MX | 120 m，≈7.5 cm/px | 8 次 |

- 航向与旁向重叠均为 **80%**
- 正射拼接：RGB 用 **Pix4Dmapper**；多光谱用 **Agisoft PhotoScan**
- 小区提取：**QGIS** + 小区矢量（shapefile，使用带缓冲的 `_buff` 版本）按小区裁剪/统计

## 4. 性状与时间点

每次飞行、每个小区产出 **32 个 RGB 植被指数（VI）+ R/G/B 波段**（多光谱为 83 个 VI + 5 个波段，本文未用）。

**时间点**：从 16 次飞行中筛出 **12 个时间点——DAP 42, 56, 61, 64, 68, 71, 75, 78, 82, 84, 90, 99**（剔除 DAP 27/34 缺失率过高的点，以及 10/127 个时间点×性状组合中 h² 过低的点）。

**性状筛选**（复现作者 `Arabidopsis_data_prep.R` 的规则，并分别对两种水分处理各做一次）：

1. 逐性状计算 12 个时间点的平均 SNP 遗传力 h²（`maize_rgb_all_traits_h2_12tp.csv`）；
2. 以基因型间距离矩阵的 **Mantel 相关**度量性状间时空相似性，相关 ≥ 0.97 建边，Louvain 模块度聚类；
3. 每个簇内保留**平均 h² 最高**的性状 → 干旱 18 个、灌溉 20 个；
4. 取两套的**交集 → 17 个性状**（`maize_rgb_selected_traits_intersection17.csv`）：

`BCC, BRVI, COM1, EBI, ExG, ExGR, G-B, G-R, GCC, MExG, MRCC, NRMBI, R-B, R/B, RCC, RGBVI, TGI`

这 17 个 RGB VI 的 12 点平均 h² 约 **0.25**，明显高于多光谱 VI 的约 0.08。

## 5. 基因型数据

- 每个 RIL 取 **8 株幼苗叶片混合**，CTAB 法提取 DNA（Chen & Ronald, 1999）
- 用 **Infinium 全基因组芯片**在 **AgReliant Genetics LLC** 分型：520 个 RIL 中 **515 个成功分型**，得到 **17,444 个 SNP**；物理位置基于 **B73 RefGEN_v3**
- **TASSEL 5** 过滤与填补：(i) 杂合位点记为缺失；(ii) 剔除 MAF < 0.05 或缺失率 > 10% 的标记；(iii) **LD-kNNi 填补**（Money et al., 2015）→ 最终 **11,334 个 SNP** 用于 GWAS 与基因组预测
- 本文使用的基因型矩阵：`data/maize_drought/SNP/geno.numerical.csv`（**533 个基因型 × 11,333 个 AgR SNP**，0/2 编码）
- **GRM**：将 0/2 转为 −1/+1 后用 `rrBLUP::A.mat`（VanRaden 方法，`min.MAF = 0.05`）构建 → **533 × 533**（对角均值 2.0，符合完全自交系），即 `data/maize_RGB/geno_GRM.csv`
- 进入分析的品系数（与折划分取交集）：**干旱 N = 522、灌溉 N = 520**

## 6. 交叉验证与归一化

- **10 个 fold-set**，每个 fold-set 把全部基因型均衡分为 6 组（1–5 训练 / 6 测试，干旱每折测试 **87** 个、灌溉 **86** 个）
- 每个 fold-set 内再做 **5 折嵌套**（4/5 训练 + 1/5 早停），测试预测取 5 个内层模型的**平均**
- 归一化：**仅用训练折**拟合 per-trait min-max，套用到训练/早停/测试；Kode、作者方法与 baseline 使用**同一归一化空间**（MSE 可直接比较）
- Baseline：**rrBLUP（GBLUP）**只预测第一个时间点，之后保持该值不变（persistence）

## 7. 田间异常事件（需在正文如实说明）

DAP **68（5/22）、71（5/25）、75（5/29）**三个飞行点上，多个 VI 出现"整体先抬升、再整体塌陷"的反常波动（如 TGI：64→68 为 +50.2、68→71 为 +40.2、71→75 为 −114.3，而其余点典型变化仅 ±3），**且所有基因型、两个水分处理同步出现**（无基因型特异性），多光谱传感器同期也测到同向变化（NDVI 5/19→5/21：0.248→0.348）。这符合短时高温→萎蔫→恢复的全田环境事件。

因此该波动在**所有方法**上都表现为那三个时间点的误差升高（iter 在 DAP 68 的 MSE = 0.0781，正常点 0.005–0.02）。我们**保留全部 12 个时间点**并如实说明，因为它反映真实田间条件下的模型表现；如需消融，可另跑"去掉 DAP 68/71/75（保留 9 点）"版本（去掉后 iter 的 MSE 从 0.01465 降到 0.0083）。

## 8. 文件清单

| 内容 | 路径 |
|---|---|
| 性状表（干旱/灌溉） | `data/maize_RGB/maize_rgb_{drought,irrigated}_traits.csv` |
| 折划分（10 fold-set） | `data/maize_RGB/maize_rgb_{drought,irrigated}_cv_10fold.csv` |
| GRM | `data/maize_RGB/geno_GRM.csv`（源：`data/maize_drought/SNP/geno.numerical.csv`） |
| 性状筛选 | `data/maize_multispectral/maize_rgb_all_traits_h2_12tp.csv`、`maize_rgb_selected_traits_{drought,irrigated}_rgb.csv`、`..._intersection17.csv` |
| 原文 | `data/maize_RGB/erad216.pdf`（Adak et al., 2023, J Exp Bot） |
| 数据说明（原始 README） | `data/_tmp_maize_rgb/README.md` |
| 结果说明 | `outputs/kode_result/maize_rgb/NOTE_maize_rgb.md` |
