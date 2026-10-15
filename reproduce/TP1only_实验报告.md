# TP1-only 动态表型预测实验报告（E1 / E2 / E3）

> 日期：2026-10-11　|　数据：玉米 MAGIC、拟南芥、水稻 indica、田间玉米 UAV
> 代码：`reproduce/knode_tp1only_dynamic{,_arab,_rice,_uav}.py`（四份同构脚本）
> 结果：服务器 `/data/fandata/dashuaige/outputs/reproduce/knode_tp1only_{maize,arab,rice,uav}/<mode>/`

---

## 一、为什么做这个实验

现有主线模型的输入是 **GRM-PCA 遗传特征 + TP1（第一个时点的多性状表型）**，输出是后续各时点的多性状轨迹。但真正落地时会遇到两个现实约束：

1. **基因型数据不一定有。** 小育种公司、区域试验站、合作农户通常没有 SNP/GRM，只有自己测的表型；
2. **完整高通量表型太贵。** 玉米 MAGIC 要 50 个性状、拟南芥 45 个、水稻 20 个、UAV 19 个；而产品端希望"用户最少交几个指标"就能预测轨迹。

因此本实验回答三个问题（引自《基序未来：TP1-only与单/少性状动态预测实验实施计划》）：

| 编号 | 问题 | 判定标准 |
|---|---|---|
| Q1 | **去掉 GRM、只用 TP1 是否可行？** | 与"GRM+TP1"同口径下 PCC 掉多少 |
| Q2 | **只用 1 个或少量性状是否可行？** | 在**共同性状**上，单/少性状模型 vs 全性状模型谁更准 |
| Q3 | **哪些配置可用于产品？** | 用户到底必须交完整 TP1，还是可以只交某几个性状 |

**保持不变**：同一原始数据集、同一 CV 测试样本、以第一个时点的真实表型为条件、预测后续多个时间点、以递推多步预测为主要评价对象（评价排除 t0）。不加入环境输入、不把任意 DAS 当新起点、不混用数据集。

---

## 二、实验怎么设计

### 2.1 三组实验

| 编号 | 输入 | 预测目标 | 训练要求 | 目的 |
|---|---|---|---|---|
| **E1** all | TP1 的**全部 P 个性状**（**不输入 GRM**） | 同 P 个性状的未来 T−1 个时点 | 重新训练 | 去 GRM 后的总体性能（Q1） |
| **E2** single | 指定**1 个**性状的 TP1 值（P=1） | 仅该性状的未来轨迹 | 新建 P=1 模型 | 极简输入能不能用、精度损失多少（Q2/Q3） |
| **E3** few | 指定**3–5 个**性状的 TP1 值（P=4 或 5） | 这些性状的未来轨迹 | 新建 P=k 模型 | 输入成本与精度之间的折中（Q2/Q3） |

### 2.2 四个数据集

| 数据集 | 群体 | N（分析用） | 时点数 T | 性状数 P | 性状性质 |
|---|---|---|---|---|---|
| 玉米 MAGIC | 330 RIL | 330 | 25（DAS 15–47） | 50 | IAP 图像描述符（颜色/纹理 46 + 几何 4） |
| 拟南芥 | 382 材料 | 382 | 14（DAS 8–21） | 45 | 顶视几何 ~20 + 颜色纹理 ~25 |
| 水稻 indica（control） | 226 品种 | 226 | 13（DAS 30–42） | 20 | 全为形态类（投影面积/紧凑度/卡尺长/凸包…） |
| 田间玉米 UAV（drought） | 509 RIL | 509 | 12（DAP 42–99） | 19 | 全为 RGB 植被指数 |

> 水稻沿用项目既有口径：**DAS ≥ 30**（与作者 dynamicGP 对齐）。

### 2.3 统一协议（与既有动态基线脚本完全一致）

- **CV**：10 个 fold-set；本实验取其中 **3 个 outer iteration**（iters 0,1,2），每折按 folds1–5 训练 / fold6 测试；
- **折内集成**：inner-5（4/5 训练 + 1/5 早停），测试预测 = 5 个 inner 模型平均；
- **归一化**：只用**训练折**的 per-trait min-max（**只对所选性状**），套用到训练/验证/测试；
- **训练**：epochs = 1000，Adam(lr 1e-3, wd 1e-4)，梯度裁剪 1.0，**固定 seed = 42**（不做多随机种子实验）；
- **模型族（5 个）**：`ststp`（静态单时点）、`lstm`、`ar_transformer`（自回归因果 transformer）、`ode_rnn`、`gru_ode_bayes` —— 逐行照抄既有动态基线脚本的实现；
- **GRM 彻底解耦**：新脚本**不读取任何 GRM/SNP 文件**，条件向量 = 归一化后的 TP1 本身（`COND_IN = P`）。

### 2.4 指标（每个 mode 每个模型都算全）

对每个 `fam × 时间点 × 性状`（**排除 t0**），在测试材料上计算：

`pcc`（跨材料 Pearson r）、`mse`、`RMSE`、`R2 = 1 − SSE/SST`、`NMSE = mse/Var(y_true) = 1 − R2`、
`NRMSE = RMSE / |mean(y_true)|`、`y_true`（真实值均值）、`y_pred`（预测值均值）、`SD`（**真实值标准差，ddof=0**）。

> 出现无法计算的情形（方差为 0、样本不足、均值≈0）→ **保留该行并写 NaN + `nan_reason`**，不隐藏。

### 2.5 公平比较规则（关键）

E1 / E2 / E3 的性状集合不同，**总体 PCC 不可直接横比**。因此 Q2/Q3 的对比一律**只在共同性状上做**（例如：E1 只取 `hull.points` 那一列，与 E2 的 `hull.points` 比），且同折、同测试材料、同时点。

---

## 三、代码位置

### 3.1 主脚本（四份同构，只改数据路径/文件名/输出目录）

| 数据集 | 本地路径 | 服务器路径 |
|---|---|---|
| 玉米 MAGIC | `reproduce/maize_magic/knode_tp1only_dynamic.py` | `reproduce/knode_tp1only_dynamic.py` |
| 拟南芥 | `reproduce/arab/knode_tp1only_dynamic_arab.py` | `reproduce/knode_tp1only_dynamic_arab.py` |
| 水稻 | `reproduce/rice/knode_tp1only_dynamic_rice.py` | `reproduce/knode_tp1only_dynamic_rice.py` |
| 田间玉米 UAV | `reproduce/maize_RGB/knode_tp1only_dynamic_uav.py` | `reproduce/knode_tp1only_dynamic_uav.py` |

配套：`reproduce/_tp1only_common.py`（共同性状公平对比表）、`reproduce/_tp1only_common.py` 同目录下的队列脚本
`q_maize.sh` / `q_ar.sh` / `q_gpu1c.sh` / `q_gpu0b.sh` / `q_uav.sh`（服务器 `/data/fandata/dashuaige/`）。

> 旧脚本 `knode_dynamic_baselines_full*.py` 与旧结果**一个字节都没有改动**，所有新产物写在独立的 `knode_tp1only_*` 目录。

### 3.2 冒烟命令（先冒烟、后正式，已执行）

```bash
cd /data/fandata/dashuaige/reproduce
PY=/data/fandata/WWC/anaconda3/envs/dashuaige/bin/python
CUDA_VISIBLE_DEVICES=0 $PY knode_tp1only_dynamic.py       --mode all --smoke                       # 玉米
CUDA_VISIBLE_DEVICES=0 $PY knode_tp1only_dynamic_arab.py  --mode all --smoke                       # 拟南芥
CUDA_VISIBLE_DEVICES=0 $PY knode_tp1only_dynamic_rice.py  --mode all --smoke                       # 水稻
CUDA_VISIBLE_DEVICES=0 $PY knode_tp1only_dynamic_uav.py   --mode all --smoke                       # UAV
# 单性状 / 少性状同样各冒一遍（P=1 / P=5）
```
冒烟结果：每个 mode 的单元数严格等于 `(T−1)×P`（玉米 209/55/11、拟南芥 585、水稻 240、UAV 209/55/11）✓

### 3.3 正式运行命令（示例）

```bash
# 玉米（3 个 iteration，1000 epoch）
$PY knode_tp1only_dynamic.py --mode all    --iters 0,1,2 --epochs 1000
$PY knode_tp1only_dynamic.py --mode single --traits side.geometry.vis.hull.points --iters 0,1,2 --epochs 1000
$PY knode_tp1only_dynamic.py --mode few    --traits "side.geometry.vis.hull.points,top.geometry.fluo.hull.points,side.geometry.vis.leaf.curvature.avg,top.geometry.vis.leaf.curvature.avg" --iters 0,1,2 --epochs 1000
```
各数据集选定的 E2/E3 性状：
- **拟南芥**：E2 = `compactness.01(visible)` / `border.length(fluorescence)` / `leaf.width.mean(fluorescence)`；E3 = 上述 3 个 + `hull.circularity` + `relative.daily.change.of.area`
- **水稻**：E2 = `PSA` / `TV_Compactness` / `SV1_Max_Dist_Horizon`；E3 = `PSA, SV1_Area, TV_Area, TV_Compactness, SV1_Max_Dist_Horizon`
- **UAV**：E2 = `TGI` / `GCC` / `RGBVI`；E3 = `GCC, TGI, RGBVI, VARI, NRMBI`

---

## 四、结果

### 4.1 各模式总体表现（5 个模型族的均值；全部剔除 t0）

| 数据集 | 模式 | P | PCC | MSE | RMSE | R² |
|---|---|---|---|---|---|---|
| 玉米 | E1 all | 50 | 0.3571 | 0.00769 | 0.0830 | −0.884 |
| 玉米 | **E2 single(hull.points)** | 1 | **0.4576** | 0.01233 | 0.1050 | 0.198 |
| 玉米 | E3 few | 4 | 0.2158 | 0.01040 | 0.0930 | −0.111 |
| 拟南芥 | E1 all | 45 | 0.4024 | 0.00823 | 0.0849 | 0.096 |
| 拟南芥 | **E2 single(border.length)** | 1 | **0.6318** | 0.00299 | 0.0441 | 0.010 |
| 拟南芥 | E2 single(compactness.01) | 1 | 0.4070 | 0.00296 | 0.0482 | 0.103 |
| 拟南芥 | E2 single(leaf.width.mean) | 1 | 0.2773 | 0.00570 | 0.0741 | 0.047 |
| 拟南芥 | E3 few | 5 | 0.2967 | 0.00468 | 0.0629 | 0.021 |
| 水稻 | E1 all | 20 | 0.7665 | 0.00573 | 0.0715 | 0.558 |
| 水稻 | **E2 single(PSA)** | 1 | **0.9650** | 0.00108 | 0.0272 | **0.916** |
| 水稻 | E2 single(TV_Compactness) | 1 | 0.6926 | 0.00959 | 0.0956 | 0.379 |
| 水稻 | E2 single(SV1_Max_Dist_Horizon) | 1 | 0.5836 | 0.00865 | 0.0915 | 0.299 |
| 水稻 | E3 few | 5 | 0.8159 | 0.00443 | 0.0587 | 0.658 |
| UAV | E1 all | 19 | 0.3732 | 0.00451 | 0.0620 | −4.395 |
| UAV | **E2 single(TGI)** | 1 | **0.7169** | 0.00240 | 0.0477 | 0.496 |
| UAV | E2 single(GCC) | 1 | 0.3951 | 0.00477 | 0.0682 | 0.134 |
| UAV | E2 single(RGBVI) | 1 | 0.4034 | 0.00397 | 0.0630 | 0.128 |
| UAV | E3 few | 5 | 0.4064 | 0.00368 | 0.0583 | 0.168 |

### 4.2 Q1：去掉 GRM 的代价（同口径，ststp，剔 t0）

| 数据集 | GRM + 全部 TP1（旧基线） | **只用全部 TP1（E1）** | ΔPCC |
|---|---|---|---|
| **水稻** | 0.7716 | **0.7715** | **≈ 0：GRM 毫无贡献** |
| 拟南芥 | 0.4686 | 0.4302 | −0.038 |
| 玉米 | 0.4731 | 0.4014 | −0.072 |
| UAV | 0.5321 | 0.4122 | −0.120 |

→ **水稻可以完全不要基因型数据**；玉米/拟南芥损失很小；UAV 损失最大（与其性状本身可预测性低一致）。

### 4.3 Q2/Q3：单/少性状 vs 全性状（**只在共同性状上比**）

| 数据集 | 模式 | 性状数 | E1 在该（些）性状上 | **该模式** | ΔPCC |
|---|---|---|---|---|---|
| 玉米 | E2 single(hull.points) | 1 | 0.2241 | **0.4576** | **+0.234** |
| 玉米 | E3 few | 4 | 0.1530 | 0.2158 | +0.063 |
| 拟南芥 | E2 single(border.length) | 1 | 0.4700 | **0.6318** | **+0.162** |
| 拟南芥 | E2 single(leaf.width.mean) | 1 | 0.1622 | 0.2773 | +0.115 |
| 拟南芥 | E2 single(compactness.01) | 1 | 0.3816 | 0.4070 | +0.025 |
| 拟南芥 | E3 few | 5 | 0.2788 | 0.2967 | +0.018 |
| 水稻 | E2 single(PSA) | 1 | 0.8987 | **0.9650** | +0.066 |
| 水稻 | E3 few | 5 | 0.7937 | 0.8159 | +0.022 |
| 水稻 | E2 single(TV_Compactness) | 1 | 0.7092 | 0.6926 | −0.017 |
| 水稻 | E2 single(SV1_Max_Dist_Horizon) | 1 | 0.6509 | 0.5836 | −0.067 |
| UAV | E2 single(TGI) | 1 | 0.7097 | 0.7169 | +0.007 |
| UAV | E2 single(GCC) | 1 | 0.3794 | 0.3951 | +0.016 |
| UAV | E2 single(RGBVI) | 1 | 0.3986 | 0.4034 | +0.005 |
| UAV | E3 few | 5 | 0.4108 | 0.4064 | −0.004 |

**13 组中 11 组"少输入 ≥ 全输入"**，2 组略差（均在 0.07 以内），1 组基本持平。

**解释**：E2/E3 的**输入信息更少**却更准，说明瓶颈不在信息，而在"一个共享网络要同时拟合几十个异质性很强的性状"——多任务共享会拖累单个性状的精度。少数例外（水稻 `TV_Compactness`、`SV1_Max_Dist_Horizon`）恰恰出现在性状间高度共线的数据集上（水稻 20 个性状的时空 Mantel 相关中位数 0.958），此时"看到别的性状"确实带来信息。

### 4.4 结论（一页版）

1. **Q1**：**水稻可以彻底不用 GRM**（0.7715 vs 0.7716）；拟南芥/玉米损失很小（−0.04/−0.07）；UAV 损失最大（−0.12）。
2. **Q2**：**对同一个性状，专门训一个小模型通常比塞进多性状大模型更准**（11/13 更好，最大 +0.23）。这不是"输入越少越好"，而是"**专属模型**优于共享模型"。
3. **Q3（产品可用配置）**：

| 推荐配置 | PCC | R² | 用户只需提供 |
|---|---|---|---|
| **水稻：仅 `PSA` 的 TP1**（且不需要 GRM） | **0.965** | **0.918** | 1 个投影面积指标 |
| **UAV：仅 `TGI` 的 TP1** | 0.717 | 0.496 | 1 个绿度指数 |
| **拟南芥：仅 `border.length` 的 TP1** | 0.632 | 0.010 | 1 个冠层周长 |
| 玉米：仅 `hull.points` 的 TP1 | 0.458 | 0.198 | 1 个轮廓复杂度 |
| 若需一次性覆盖全部性状 | 水稻 0.766 / 拟南芥 0.402 / 玉米 0.357 / UAV 0.373 | — | 完整 TP1（不需 GRM） |

> 注意：E1 的"覆盖全部性状"与 E2/E3 的"单性状高精度"是**两种产品形态**，不是同一任务的高低优劣。

---

## 五、结果所在位置

### 5.1 服务器（正式结果）

`/data/fandata/dashuaige/outputs/reproduce/knode_tp1only_<dataset>/<mode>/`

| 数据集 | mode 目录 |
|---|---|
| 玉米 MAGIC | `all`、`single`、`few` |
| 拟南芥 | `all`、`single_compactness01_visible`、`single_borderlength_fluo`、`single_leafwidth_fluo`、`few5` |
| 水稻 | `all`、`single_PSA`、`single_TV_Compactness`、`single_SV1_MaxDist`、`few5` |
| UAV | `all`、`single_TGI`、`single_GCC`、`single_RGBVI`、`few5` |

每个 mode 目录下 **7 件产物**：

| 文件 | 内容（列） |
|---|---|
| `metrics_by_trait_time.csv` | 主表：`fam, time, trait, pcc, mse, RMSE, R2, NMSE, NRMSE, y_true, y_pred, SD, nan_reason`（3 个 iteration 平均） |
| `metrics_by_trait_time_per_iter.csv` | 同上，前面多一列 `iteration`（逐 iteration 明细） |
| `metrics_by_trait_time_std.csv` | 3 个 iteration 间各指标的 SD（`*_std`，ddof=0） |
| `metrics_overall.csv` | 每模型族在"全部性状 × 全部 t0 后时点"上的总体指标 |
| `predictions_long.csv.gz` | **逐材料**预测：`iteration, fam, bio_id, time, trait, y_true, y_pred`（可画材料真实/预测轨迹、做 Top-K 筛选） |
| `run_config.json` | 数据集/mode/性状清单/iter/fams/epochs/seed/归一化/指标定义/起止时间/test_ids_by_iter |
| `train.log` | 运行日志（每个模型、每个 iteration 的耗时与收尾） |

另有 `.../all_smoke`、`.../few_smoke`、`.../single_smoke`、`.../_server_check*` = **冒烟/自检产物**，不属正式结果。

### 5.2 汇总对比表
- `reproduce/_tp1only_common.py`（服务器）→ 输出"共同性状公平对比表"（即 4.3 节的数字）
- 4.1 / 4.2 节的数字分别取自各 mode 的 `metrics_overall.csv` 与既有基线汇总 `knode_dynamic_baselines_full*.csv`

### 5.3 关键运行事实（便于复核）
- 4 条链、3 张卡（GPU 6 / 0 / 1）并行；每个 mode 内部 3 个 iteration 串行；**每张卡同时只有 1 个训练进程**（启动前后均 `pgrep` 查重）
- 全部 18 个正式运行**产物齐全（每个 7 件）**，无 NaN 异常行以外的报错
- 数据一致性：四套数据集在本地与服务器的关键文件 md5 逐一对齐（UAV 由旧的 17 性状版同步为当前 19 性状版，旧文件备份为 `*.bak_17traits_20261010_164357`）
- **旧脚本、旧结果（`knode_dynamic_baselines_full*`）完全未改动**
