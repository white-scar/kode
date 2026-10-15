# 水稻（indica salt panel）数据集说明

> 用于 Kode/dynamicGP 复现的水稻表型-基因型数据。所有数字均由 `data/rice_salinity/` 下的原始文件核对得到。

## 1. 群体与试验

数据来自 **Al-Tamimi et al. (2016, *Nature Communications*)** 报道的 2015 年水稻盐胁迫实验，于澳大利亚阿德莱德 **Plant Accelerator（Australian Plant Phenomics Facility）** 完成，采用 LemnaTec Scanalyzer 3D 成像平台。该报道包含两个独立实验：**0254 = indica 群体**、**0255 = aus 群体**；本文使用 **0254 indica 群体**。

试验中每个基因型同时在**对照（Control）与盐处理（Salt）**下种植于载苗车（cart），两个处理配对排列，并分布在 **两个 Smarthouse（NE / NW）** 中，每个处理在对应 Smarthouse 内约 2 个重复车次。原始成像记录共 **14,784 条**（= 1,056 个车次 × 14 个成像日；NE/NW × Control/Salt 各 3,696 条），涉及 **257 个基因型**。

## 2. 成像与性状

自播种后第 28 天起**逐日成像**（2015-03-11 至 2015-03-25），每次采集同时获得**两个侧视图（SV1、SV2）与一个顶视图（TV）**的 RGB 图像。

从图像中提取 **20 个形态性状**（对应原始文件的 `RGB_*` 与 `Projected Shoot Area pixels` 列）：

| 组 | 个数 | 性状 |
|---|---|---|
| **PSA** | 1 | Projected Shoot Area（投影冠层面积，pixels） |
| **SV1**（侧视 1） | 7 | Area、Boundary Points-to-Area Ratio、Caliper Length、Center of Mass Y、Compactness、Convex Hull Area、Max Dist Above Horizon Line |
| **SV2**（侧视 2） | 7 | 同上 7 个 |
| **TV**（顶视） | 5 | Area、Boundary Points-to-Area Ratio、Caliper Length、Compactness、Convex Hull Area |

即 **1 + 7 + 7 + 5 = 20**。此外每个车次记录**失水量**（Weight Before/After、Water Amount），作者据此计算蒸腾速率与 TUE 等衍生性状（本文未使用）。

## 3. 时间点

共保留 **14 个成像日：DAS 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42**（缺 DAS 29）。

## 4. 表型处理与本文使用的表

作者的完整表型流程使用 R 包 `imageData`（Brien, 2016）做数据清洗、PSA 自然三次样条平滑、AGR/RGR/SIIT 等衍生性状计算，并用混合线性模型（ASReml）产出**空间校正的 BLUE**；其 BLUE 工作表（`Rice2015.0254.v8.xlsx`、`0254 Cart BLUEs.v8.csv`）覆盖的是作者自行派生的性状（Area.smooth 及各种 AGR/RGR/TUE/Transpiration），**不含本文使用的 20 个原始 LemnaTec 描述符**。

本文使用的表型表（`rice_indica_control_traits.csv` / `rice_indica_salt_traits.csv`）给出 **226 个基因型 × 14 个成像日 × 20 个性状**，对照与盐处理各一套；数值为**每个基因型在每个成像日、每个处理下的品系值**（由重复车次汇总，取值形态与"重复车均值"一致）。

> ⚠️ 待确认：这批 20 性状的"品系 × 日期"值究竟是**重复车直接平均**还是**经混合模型空间校正的 BLUE**——从数值形态看更像前者（约 64% 整数、36% 为 .5 结尾，符合 2 个整数取均值）。若论文要写"Mixed-model BLUE"，需向数据提供方确认。

## 5. 基因型数据

- **SNP**：HDRA 700K 芯片（Cornell，McCouch 实验室）；经 **MAF ≥ 0.05** 与缺失位点过滤后，indica 面板保留 **19,436 个 SNP** × **228 个品种**（`Hapmap_IND700.maf005.nomissing.hmp.txt`）。
- **亲缘/基因组关系矩阵**：作者用 **TASSEL** 计算（`Kin_IND700.maf005.nomissing.txt`，**228 × 228**）；本文取**同时具有表型与基因型的 226 个品种**，得到 **226 × 226** 的 K（`rice_GRM_226x226.csv`，与 TASSEL K 差异仅 2×10⁻⁶，为 CSV 舍入）。
- **交叉验证**：10 个 fold-set，每个 fold-set 将 226 个品种随机分为 6 组（1–5 训练 / 6 测试，每个测试集 37 个品种）；每个 fold-set 内再做 5 折嵌套用于集成/早停。

## 6. 性状冗余与筛选（本次分析）

复现作者的性状聚类流程（Mantel 相关 + 模块度聚类 + 簇内取遗传力最高者）：

- **遗传力**：以 `rrBLUP::mixed.solve(y, K)` 做 GREML，逐 `性状 × DAS` 估计 h² = Vu/(Vu+Ve)，再对 14 个 DAS 取平均。**20 个性状 h² = 0.25–0.53**（最高 TV_Compactness 0.531、TV_Boundary_Ratio 0.522；最低 SV1_Boundary_Ratio 0.250），无一为噪声性状。
- **Mantel 相关**（性状间时空距离矩阵的相关）：中位数 **0.958**，93/190 对 ≥0.96、30/190 对 ≥0.99 → 性状间高度冗余。
- **聚类结果**：阈值 0.96 → 2 簇；阈值 0.97–0.99 → **3 个时空簇**：
  - 尺寸类（14 个性状：PSA / Area / Caliper Length / Convex Hull Area / Center_Y / Max_Dist_Horizon）
  - 轮廓比（3 个：SV1/SV2/TV Boundary_Ratio）
  - 紧实度（3 个：SV1/SV2/TV Compactness）
  - 阈值曲线（within/out 比值）在 0.95–0.96 达到最大。
- 代表性性状：**SV1_Area、TV_Boundary_Ratio、TV_Compactness**（3 簇各取 h² 最高者）。

**本文决定保留全部 20 个性状、不做预筛选**，理由：(i) 20 个性状 h² 全部中等以上（0.25–0.53），没有需要剔除的低质量性状（作者的筛选动机是从 355 个性状中剔除冗余/低质量）；(ii) 两类方法均建立在 **rank-2 时空算子**上，性状冗余由 SVD 截断吸收，不会造成秩膨胀；(iii) 不筛选可避免选择偏差，并保证两类方法在同一性状空间上比较。作为敏感性分析，将评价限制到 3 个代表性状后所有结论不变（recursive 场景下作者方法 MSE 为 Kode 的 5.8×，全性状时为 11.9×；iterative 场景 17.0× vs 29.1×）。

## 7. 归一化

所有性状在**全部品系与时间点**上做 min-max 归一化；Kode 与作者方法使用**同一套**归一化空间，因此 MSE 可直接比较。
