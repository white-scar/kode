# ST-STP 单性状 vs 全性状（拟南芥）逐性状对照

> 数据：拟南芥 同一份表型、同一 CV、iters=0,1,2、inner-5、seed=42、只用 TP1（无 GRM）；仅 ST-STP。
> 比较：**逐性状一一匹配**（同一 trial 的 13 个未来真实观测时点），不拿单性状 PCC 去比全 45 性状的总体均值。
> 持平阈值：|ΔPCC| < 0.01（描述性，不作显著性检验；3 个 iteration 不足以声称统计显著）。

## 总体（45 个性状）

| 判定 | 数量 | 占比 |
|---|---|---|
| 提升 (ΔPCC > +0.01) | 16 | 36% |
| 持平 (|ΔPCC| < 0.01) | 10 | 22% |
| 下降 (ΔPCC < -0.01) | 19 | 42% |

- ΔPCC：均值 -0.0173，中位 -0.0054，范围 -0.3592 ~ +0.1930
- ΔMSE（越低越好）：均值 +0.00006，中位 -0.00005；ΔMSE<0 的性状 25 个
- 后期（最后 4 个时点）ΔPCC 均值 -0.0701；其中改善的性状 13 个

## 提升最多（前 8）

| trait | PCC_single | PCC_all | ΔPCC | ΔMSE | 判定 |
|---|---|---|---|---|---|
| `hsv.s.mean..color.related.trait.based.on.visible.light.top.view.` | 0.5128 | 0.3197 | +0.1930 | -0.00087 | 提升 |
| `hull.points..geometry.trait.based.on.fluorescence.top.view.` | 0.3182 | 0.1867 | +0.1315 | -0.00015 | 提升 |
| `leaf.width.mean..geometry.trait.based.on.fluorescence.top.view...px.` | 0.2805 | 0.1724 | +0.1080 | -0.00028 | 提升 |
| `hull.fillgrade..geometry.trait.based.on.visible.light.top.view.....` | 0.3912 | 0.3324 | +0.0588 | -0.00031 | 提升 |
| `cog.x..geometry.trait.based.on.visible.light.top.view...px.` | 0.6348 | 0.5800 | +0.0548 | -0.00089 | 提升 |
| `cog.x..geometry.trait.based.on.fluorescence.top.view...px.` | 0.6331 | 0.5796 | +0.0535 | -0.00089 | 提升 |
| `hull.circularity..geometry.trait.based.on.fluorescence.top.view.` | 0.2747 | 0.2242 | +0.0506 | -0.00063 | 提升 |
| `hull.centroid.x..geometry.trait.based.on.visible.light.top.view...px.` | 0.5987 | 0.5554 | +0.0433 | -0.00085 | 提升 |

## 下降最多（后 8）

| trait | PCC_single | PCC_all | ΔPCC | ΔMSE | 判定 |
|---|---|---|---|---|---|
| `hsv.h.mean..color.related.trait.based.on.visible.light.top.view.` | 0.2334 | 0.5926 | -0.3592 | +0.00485 | 下降 |
| `daily.log.change.rate.of.area..geometry.trait.based.on.fluorescence.top.view...relative.day.` | 0.0382 | 0.2025 | -0.1643 | +0.00004 | 下降 |
| `relative.daily.change.of.area..geometry.trait.based.on.fluorescence.top.view...relative.day.` | 0.0582 | 0.2112 | -0.1530 | +0.00012 | 下降 |
| `leaf.count..geometry.trait.based.on.fluorescence.top.view.` | 0.0114 | 0.1425 | -0.1311 | +0.00008 | 下降 |
| `lab.b.skewness..color.related.trait.based.on.fluorescence.top.view.` | 0.2960 | 0.4138 | -0.1178 | +0.00128 | 下降 |
| `percentage.of.daily.change.of.leaf.length.sum..geometry.trait.based.on.fluorescence.top.view...percent.change.day.` | 0.0061 | 0.1145 | -0.1083 | +0.00002 | 下降 |
| `hsv.h.skewness..color.related.trait.based.on.visible.light.top.view.` | 0.4518 | 0.5527 | -0.1009 | +0.00092 | 下降 |
| `hsv.dgci.mean..color.related.trait.based.on.visible.light.top.view.` | 0.5419 | 0.6281 | -0.0862 | +0.00215 | 下降 |

## 备注

- 单性状模型与 E1 all 使用**相同的 outer test IDs、相同 13 个未来时点、相同训练折 min-max**（各自性状的 min-max）。
- `ststp` 为静态多时点预测头（TP1 → 全部时点一次读出），不是逐步递归动力学模型。
- 明细见 `comparison_single_vs_all_ststp_by_trait_time.csv`（585 行）；逐性状汇总见 `..._by_trait.csv`（45 行）。
