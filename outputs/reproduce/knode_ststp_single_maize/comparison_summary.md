# ST-STP 单性状 vs 全性状（玉米 MAGIC）逐性状对照

> 数据：玉米 MAGIC 同一份表型、同一 CV、iters=0,1,2、inner-5、seed=42、只用 TP1（无 GRM）；仅 ST-STP。
> 比较：**逐性状一一匹配**（同一 trial 的 24 个未来真实观测时点），不拿单性状 PCC 去比全 50 性状的总体均值。
> 持平阈值：|ΔPCC| < 0.01（描述性，不作显著性检验；3 个 iteration 不足以声称统计显著）。

## 总体（50 个性状）

| 判定 | 数量 | 占比 |
|---|---|---|
| 提升 (ΔPCC > +0.01) | 14 | 28% |
| 持平 (|ΔPCC| < 0.01) | 2 | 4% |
| 下降 (ΔPCC < -0.01) | 34 | 68% |

- ΔPCC：均值 -0.0611，中位 -0.0462，范围 -0.3941 ~ +0.2018
- ΔMSE（越低越好）：均值 +0.00014，中位 +0.00009；ΔMSE<0 的性状 19 个
- 后期（最后 6 个时点）ΔPCC 均值 -0.1121；其中改善的性状 9 个

## 提升最多（前 8）

| trait | PCC_single | PCC_all | ΔPCC | ΔMSE | 判定 |
|---|---|---|---|---|---|
| `side.geometry.vis.hull.points` | 0.4565 | 0.2548 | +0.2018 | -0.00186 | 提升 |
| `top.intensity.fluo.hsv.v.skewness` | 0.2126 | 0.1267 | +0.0859 | -0.00034 | 提升 |
| `side.geometry.vis.leaf.curvature.avg` | 0.2753 | 0.2083 | +0.0669 | -0.00013 | 提升 |
| `top.geometry.fluo.hull.points` | 0.1991 | 0.1465 | +0.0526 | +0.00006 | 提升 |
| `top.intensity.vis.hsv.h.histogram.s_avg.bin.07.076_089` | 0.3479 | 0.2956 | +0.0523 | -0.00007 | 提升 |
| `top.intensity.vis.hsv.h.histogram.s_avg.bin.11.127_140` | 0.0507 | 0.0135 | +0.0372 | -0.00009 | 提升 |
| `top.intensity.vis.hsv.normalized.h.histogram.bin.13.153_165` | 0.1432 | 0.1088 | +0.0344 | -0.00020 | 提升 |
| `side.texture.fluo.lab.a.std` | 0.6493 | 0.6160 | +0.0333 | -0.00045 | 提升 |

## 下降最多（后 8）

| trait | PCC_single | PCC_all | ΔPCC | ΔMSE | 判定 |
|---|---|---|---|---|---|
| `top.intensity.vis.hsv.normalized.v.histogram.bin.03.025_038` | -0.0262 | 0.3679 | -0.3941 | +0.00011 | 下降 |
| `top.intensity.vis.hsv.normalized.v.histogram.bin.07.076_089` | 0.1282 | 0.4132 | -0.2850 | +0.00036 | 下降 |
| `top.intensity.vis.hsv.h.skewness` | 0.2502 | 0.5117 | -0.2615 | +0.00196 | 下降 |
| `top.intensity.vis.hsv.v.histogram.s_avg.bin.07.076_089` | 0.1447 | 0.3887 | -0.2440 | +0.00120 | 下降 |
| `top.texture.fluo.hsv.s.mean` | 0.0664 | 0.2857 | -0.2193 | +0.00014 | 下降 |
| `top.intensity.vis.hsv.h.histogram.s_avg.bin.04.038_051` | 0.2330 | 0.4246 | -0.1916 | +0.00083 | 下降 |
| `side.texture.vis.lab.a.mean` | 0.2452 | 0.4142 | -0.1690 | +0.00040 | 下降 |
| `top.intensity.vis.hsv.v.histogram.s_avg.bin.06.063_076` | 0.3355 | 0.4864 | -0.1509 | +0.00044 | 下降 |

## 备注

- 单性状模型与 E1 all 使用**相同的 outer test IDs、相同 24 个未来时点、相同训练折 min-max**（各自性状的 min-max）。
- `ststp` 为静态多时点预测头（TP1 → 全部时点一次读出），不是逐步递归动力学模型。
- 明细见 `comparison_single_vs_all_ststp_by_trait_time.csv`（1200 行）；逐性状汇总见 `..._by_trait.csv`（50 行）。
