# 图三（信息源消融 / cond ladder）正文草稿 + 数据核对

> 数据: `outputs/reproduce/cond_ladder_{arab_dg30_r2, rice_dg10_r2, rgb_dg50_r2}/`
> 图:   `pic-code/pic3/{arab,rice,rgb}/fig_cond_ladder_*.svg`  (只出 SVG)
> 口径: 10 fold-set, inner-5 集成, fold6 测试, 训练折 per-trait min-max, 评估剔除 t1; rank=2, 锁定 D_G。

## 1. 结论（两句）

**遗传背景和早期表型均能够提供关于后续发育动力学的信息，但两者的相对贡献及互补程度取决于早期表型状态已经包含了多少未来发育信息。**

**GRM 的边际贡献具有明显的 context dependence：在 TP1 高度信息化的体系中有限，在 TP1 信息不足的田间 UAV 场景中则很大。**

## 2. 图说（用图三里的五个档名）

信息源消融显示，早期表型与基因组信息对未来发育预测的相对贡献在不同表型体系中存在明显差异。
在拟南芥中，**TP1-conditioned** 已达 PCC 0.482，而加入 GRM 后 **GRM+TP1-conditioned Kode** 仅提高至 0.485，
表明基因组信息在早期多性状状态基础上的额外增益有限。
水稻中的这一现象更加明显：**TP1-conditioned** 达到 0.812，高于 **GRM+TP1-conditioned Kode** 的 0.788；
与此同时，**GRM-conditioned** 本身仍取得 0.768 的较高精度，说明该结果并非源于基因组信息缺乏预测能力，
而更可能反映：**在该体系中 t1 已经几乎确定了后续轨迹（S_j≈0.76），因此基因组信息所携带的未来信息与之条件重叠**。
相比之下，田间 UAV 数据中 **TP1-conditioned** 仅为 0.344，而 **GRM-conditioned** 达到 0.378，
二者联合后 **GRM+TP1-conditioned Kode** 提高至 0.440，显示出明显的信息互补。

### 五个档的新名字（图里已改，图注里请照此写）

| 数据里的 key | 图上的名字 | 含义 |
|---|---|---|
| `persistence` | **TP1 persistence** | 把 t1 的观测值直接沿用（不做动力学） |
| `shared` | **Shared dynamics** | 不给个体信息（全 0 输入），只学一套共同动力学 |
| `grm` | **GRM-conditioned** | 只用 GRM PCA（+ 共享动力学）生成个体算子 |
| `tp1` | **TP1-conditioned** | 只用 t1 表型（+ 共享动力学）生成个体算子 |
| `full` | **GRM+TP1-conditioned Kode** | 两者都给的完整 Kode（=主方法 rec） |

## 3. 数字核对（都来自新跑的 cond ladder）

| 图上名字 | arab (30,2) | rice (10,2) | uav (50,2) |
|---|---|---|---|
| TP1 persistence | 0.4066 | 0.7529 | 0.3696 |
| Shared dynamics | 0.4171 | 0.7796 | 0.3422 |
| GRM-conditioned | 0.4338 | 0.7684 | 0.3778 |
| TP1-conditioned | 0.4824 | **0.8123** | 0.3438 |
| GRM+TP1-conditioned Kode | **0.4854** | 0.7882 | **0.4398** |

支撑"互补程度"的一句话量化（`Full − max(单信息源)`）：

| 数据集 | Full − TP1 | Full − max(TP1, GRM) | 解读 |
|---|---|---|---|
| arab | +0.0031 | +0.0031 | 边际增益有限 |
| rice | −0.0241 | −0.0241 | 负贡献（条件重叠） |
| uav | +0.0960 | **+0.0620** | 明显互补 |

交叉验证：`GRM+TP1-conditioned Kode` 与独立跑的 Full Kode `D_G x rank` 联合扫描在 (30,2)/(10,2)/(50,2) 上
分别 = 0.48544 / 0.78816 / 0.43983，与 cond ladder 的 `full` 档**逐位相同**。

## 4. TP1 的信息含量（替代"性状重复"的说法）

| 数据集 | P | N | T | t1 | 平均 \|r\|(t1) | PC1 解释率 | effective rank | S_j 均值 | 持续性基线 PCC |
|---|---|---|---|---|---|---|---|---|---|
| arab | 45 | 382 | 14 | DAS 8 | 0.222 | 26.4% | 8.67 / 45 | 0.408 | 0.4066 |
| rice | 20 | 226 | 13 | DAS 30 | 0.352 | 36.3% | 4.44 / 20 | **0.755** | 0.7529 |
| uav | 19 | 509 | 12 | DAS 42 | 0.448 | 42.6% | 2.91 / 19 | **0.382** | 0.3696 |

`S_j = mean_{t>1} |corr(y_j(t1), y_j(t))|`（逐性状，再跨性状平均）。数据: `outputs/reproduce/tp1_info_content_summary.csv`。

**要点（重要）**
- 不要用"性状重复/高度相关"解释水稻：冗余度指标反而显示 **UAV 的 TP1 最冗余**（effective rank 2.9/19），
  水稻居中（4.44/20），拟南芥最不冗余（8.67/45）。用性状相关解释水稻会与数据自相矛盾。
- 该用"**条件重叠**"：水稻 `S_j≈0.76`，即 t1 已几乎确定后续轨迹；UAV `S_j≈0.38`，t1 远不足。
  跨数据集正好对应 GRM 边际贡献的排序（rice −0.024 < arab +0.003 < uav +0.096）。
- 逐 trait 的 `corr(S_j, Δ)`：arab −0.31 (p=0.036)、rice +0.68 (p=0.001)、uav −0.22 (p=0.37)，
  **组内方向不一致且 n 小，建议正文只讲跨数据集**，组内相关放补充材料。
- UAV 最强的一格：19 个性状**全部** Δ>0（0/19 为负），是唯一无条件互补的体系。

数据: `outputs/reproduce/tp1_info_content_per_trait.csv`（逐 trait 的 S_j / 三个 cond 的 PCC / Δ）。
