# KNode / Kode 项目硬性规则（每次实验必须遵守）

> 这些规则来自用户多次强调的要求。写在文件里是为了防止长对话上下文压缩后丢失。
> 与用户口头要求冲突时，以用户最新要求为准。

## 0. 结果输出：缺一不可

- 每个实验必须同时输出 **PCC、MSE、NRMSE（或 NMSE）、R²**，不允许只算 PCC。
- 必须分三层输出：`overall`（pooled）、`by_time`、`by_trait`（适用时）。
- 交付格式统一为 **CSV**，最终结果不允许只给 json/npz。
- 图片只输出 **SVG**，不要 PNG / PDF（用户 2026-09-12 要求）。
- 作者方法、本方法、基线三者必须**同口径、同字段、同归一化空间**，MSE 才能直接比较。

## 1. 默认协议（除非用户明确要求另开一套）

- 归一化：只用**训练折**的 per-trait min-max，套用到验证/测试；验证/测试不参与任何统计量。
- 集成：inner 5 折（4/5 训练 + 1/5 早停），test 预测取 5 折平均。
- 十折：10 个 fold-set，fold1–5 训练 / fold6 测试。
- 作者复现：tp1 + MegaLMM + dynamicGP；DMD 缺口用参数 `time_gaps`（玉米 in-window=4、CV3=3，拟南芥=0）。
- 所有方法在同一归一化空间计算 MSE，禁止各算各的归一化。

## 2. 冒烟测试：长跑之前强制

- 先跑 1 fold + 1 iteration（或 `--smoke`）。
- 检查：脚本跑通、输出列齐全（PCC/MSE/...）、无 NaN、无泄漏、路径正确。
- 冒烟通过才允许全量运行。禁止上来直接跑十折/全量。

## 3. 后台运行，不盯屏

- 一律后台：`nohup ... > log 2>&1 &`，或走 `reproduce\server_ssh.py` 后台。
- 启动后立即汇报：脚本路径、日志路径、PID、预计时长。
- **不要反复轮询等待**；只在完成、报错、或用户询问时汇报。

## 4. 完成前自检清单

- [ ] PCC 和 MSE 都算了
- [ ] overall + by_time + by_trait 都有
- [ ] 训练折 scaler，无泄漏
- [ ] 10 个 fold-set
- [ ] inner-5 集成
- [ ] 作者 / 本方法 / 基线同口径
- [ ] 输出为 CSV
- [ ] 结果 + 运行 metadata 一起保存

## 5. 汇报结果的固定格式

`实验名 | 脚本路径 | 数据路径 | PCC | MSE | NRMSE | R² | 结果路径`

数字必须来自文件，不许凭记忆或估算。

## 6. 服务器约束

- 只在 `/data/fandata/dashuaige/` 内生成、上传、运行，尽量用相对路径。
- 最多用 2 张 GPU，且必须是空闲卡；CPU 任务不得影响他人。
- 新建虚拟环境要记录并告知用户，方便下次交接。

## 7. 任务清单（待办）

### 拟南芥外推重跑 + 旧文件清理
- [ ] 1. 冒烟测试：`reproduce/knode_extrap_maize_arab_v2.py --dataset arab`（1 fold + 少 epoch），确认能跑通
- [ ] 2. 后台全量跑拟南芥外推（十折，主方法 rank2+skip+affine），输出到 `outputs/reproduce/knode_extrap_maize_arab/`
- [ ] 3. 更新 `reproduce/pic-code/new_fig11_fig12.py` 第 108 行 `ak`：旧 `knode_hyper_ode_arab_all_detail_rank2.csv` → 新 `outputs/kode_result/arab/knode_arab_tp1skip_ode_hyper_skip_per_trait.csv`，重画 fig5b
- [ ] 4. 核对/重做补充图 E8-ara（`reproduce/pic-code/make_fig4_arab_data.py` 仍在读旧 per_trait_rank2）
- [ ] 5. 确认图更新后再删除以下 4 个文件：
      - `outputs/reproduce/knode_hyper_ode_arab_all_detail_rank2.csv`（现喂 fig5b）
      - `outputs/reproduce/knode_hyper_ode_hyper_arab_per_trait_rank2.csv`（现喂 E8-ara）
      - `outputs/reproduce/knode_hyper_iter_hyper_arab_per_trait_rank2.csv`（现喂 E8-ara）
      - `reproduce/knode_hyper_ode_arab_affine.py`（全项目零引用）


## 8. 【铁律】用户给的脚本 = 模板，必须原样照抄（2026-09-14 用户严厉强调，绝不可再犯）

用户给一个脚本并说"参考/照着这个"，含义是：**该脚本怎么写，你就怎么写，只允许换数据集**。

- **必须逐行照抄原脚本**：模型定义、训练循环、inner 集成、指标计算、归一化、聚合逻辑、输出文件名与列名，全部与原脚本一致。
- **只允许修改**：数据路径/文件名、数据集名称、与数据集相关的维度常量（N/P/T、GRM 维度）、输出目录名。
- **严禁（任何理由都不行）**：
  - 自己重新设计输出格式；
  - 删掉模板原有的输出文件；
  - 把模板的"逐 fold/逐 trait 原始长表"简化成"汇总表"；
  - 改写聚合口径、重命名列、精简代码结构。
- **验收标准**：新脚本产出的文件**数量、文件名、列名**必须与模板逐一对应。
  例：模板输出 `*_per_trait.csv`（含 `foldset x timepoint x trait` 的原始长表），新脚本也必须输出同样格式的表。
- 若认为模板某处有问题：**先问用户，禁止擅自修改**。
- 反面案例（教训）：写 `knode_maize_rgb_np_tp1_skip_affine.py` 时，未照抄模板、自创 `by_time/by_trait/overall/per_foldset` 汇总输出，丢掉了模板原有的 per-trait 原始长表，
  导致逐 (fold x DAS x trait) 的 MSE 没有落盘且无法反推，无法画出与 pic2 一致的箱线图，只能浪费一次重跑。


## 9. 【铁律】启动后台任务前必须先查重（2026-09-18 教训）

**事故**：玉米 / UAV / 水稻的 rank 脚本被重复启动（早上一次、中午又一次），结果 52 个进程（应有 30）、
显存占到 97%、GPU 利用率却只有 31% —— 同一批组合被算两遍，白烧算力，而且是用户先发现的。

**启动前必做（按顺序）**：

1. `pgrep -fc <脚本名>`：不为 0 说明已有实例在跑，**禁止重复启动**；
   先看上一轮日志尾部（是否已 `DONE` / `aggregated`），确认那批是已完成还是仍在跑。
2. 看日志时间戳：脚本开头 `: > log` 会清空日志，所以日志里的时间就是**最后一次启动**的时间。
3. 启动后立刻 `pgrep -fc` **复查进程数是否符合预期**（pass1 = 1 个；10 折并行 = 10 个；
   两环境串行脚本同一时刻只应有一个环境在跑）。数量不对立即排查，别放着不管。

**重启的正确姿势**：先全杀 → `pgrep` 确认 0 → 再启动一次。
禁止在不杀旧实例的情况下再启动一遍。

**清理缓存的代价**：删 `uv2*` 会让已完成的组合全部重算，清理前先想清楚值不值得，
宁可只删确定作废的部分。

**长任务要顺手看资源**：`nvidia-smi`（显存/利用率）+ `cat /proc/loadavg`（CPU）。
显存 > 90% 或利用率与进程数严重不匹配（如 52 进程却只有 31%）时，先怀疑重复启动。


## 10. 结论表述规范：写「低秩」，不写「最优 rank = 2」（2026-09-18 用户明确要求）

**规范**：rank 分析的结论是**低秩饱和（low-rank saturation）**，不是"最优 rank = 2"。

- 依据（玉米新口径 10 折，full 条件）：r=1 → 0.5050、r=2 → 0.5127、r=3 → 0.5033、r=4 → 0.5024、
  r=8 → 0.5038、r=20 → 0.5049、r=50 → 0.5051 —— **r ≥ 2 全部落在 0.502–0.507（极差 < 0.011）**，
  是**平台**而不是尖峰。
- 因此 r = 2 / 3 / 4 / 8 的结论一致；写成唯一点会被质疑"为什么不是 3"，而两者实际没有差别。
- 图标题写 "saturates at low rank"（pic4/fig_rank.py 已经是这样），正文写 "low-rank saturation"。
- **同理适用于其他超参**：`GRM PCA dim`、`hid`、`depth`、`act` 等一律按"稳健平台 + 取平台内代表值"表述，
  不写"唯一最优"；选代表值的理由写成"位于稳定区间中部"，并给出区间内差异的量级。

## 11. 【铁律】主模型必须 per-foldset 重置随机种子（2026-09-21 修复，必须保留）

**背景**：四个主模型脚本（`maize_magic/knode_maize_np_tp1_affine.py`、`arab/knode_arab_np_tp1_affine.py`、
`rice/knode_rice_np_tp1_affine.py`、`maize_RGB/knode_maize_rgb_drought_np_tp1_affine.py`）原来只在
模块级设一次 `torch.manual_seed(42); np.random.seed(42)`，循环是 `for mode → for foldset → for inner_fold`，
RNG 一路流下去；而 cond ladder（`knode_cond_ladder_*.py`）在**每个 (fold, cond) 前都重置**到 42。
后果：只有 foldset 1 两条产线初始化相同，foldsets 2–10 全都不同 —— 图三/图四 与 图二/图五/图六对不上。

**实测量级**（修复前，主模型 − cond ladder 的 PCC 均值差）：
玉米 −0.0005、拟南芥 +0.0005、水稻 +0.0003、**无人机 +0.0086（相对偏差 1.94%）**。
fold 级指纹：**四个数据集的 fold 1 差异都是 0.0000**，其余 9 折随机正负。
无人机之所以最大：折间波动最大（SD 0.0327）+ 10 折差异 7 正 3 负、不抵消。

**修复**：主脚本新增模块级 `SEED = 42` 与 `--seed` 参数，在 foldset 循环开头加
```python
torch.manual_seed(SEED); np.random.seed(SEED)
```
`knode_hyperparam_ablation.py`（包装器）改为 `base.SEED = args.seed`，并把 `'--seed', str(args.seed)`
传进 `sys.argv`。**绝不能硬编码 42** —— 否则 `seed1234/2024/2025` 三档会全部塌成同一个种子。

**为什么对齐就能一致**：`CondLadderModel` 与 `HyperODEModel` 逐层等价（`hyper_enc`/`hyper_U`/`hyper_V`/
`hyper_b` 同序同形，`forward` 同），训练循环也不消耗 RNG（无 dropout、无 shuffle，数据顺序固定）——
**全局 RNG 的唯一用途是 `nn.Linear` 初始化**，所以 RNG 状态对齐后两边应逐位相等，
残差只剩 GPU 非确定性（实测 ~5e-5）。

**验收标准**：任意数据集，主模型 `ode_hyper` 的 per-foldset PCC 应与 cond ladder `full` 的
per-foldset PCC 逐折吻合（残差 < 0.003）。修复后实测：无人机 fold2 0.4061→0.3899（基准 0.3879）。

**当前一致性水平（2026-09-21 修复并同步后）**：四个数据集 × 图二/三/四/五/六，
rec PCC / iter PCC / rec MSE / iter MSE 的相对偏差**全部 < 0.4%**（全局最大 0.382%，玉米 rec MSE）。
