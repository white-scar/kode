# dynamicGP_rrbench_legacy —— 从已删除的 `outputs/dynamicGP/` 里抢救出来的 2 个文件

## 这里有什么

| 文件 | 来源（已删除） | 大小 |
|---|---|---|
| `rrBLUP_benchmarks_prediction_accuracies_maize.csv` | `outputs/dynamicGP/Training_Validation_Results/rrBLUP_benchmarks_prediction_accuracies.csv` | 1262.8 KB |
| `rrBLUP_benchmarks_prediction_accuracies_arabidopsis.csv` | `outputs/dynamicGP/Arabidopsis/Training_Validation_Results/rrBLUP_benchmarks_prediction_accuracies.csv` | 923.4 KB |

列：玉米 `Method, Iteration, Trait, Time, Accuracy, MSE`；拟南芥 `Iteration, Trait, Time, Accuracy, MSE, R2, NRMSE`。

## 谁在用

`reproduce/pic-code/E3-4/make_table2_data.py`（以及同内容副本 `reproduce/pic-drawing/pic-code/table2/make_table2_data.py`）
→ 生成 E-Table2 的 RR-BLUP benchmark sweep 溯源（`rrblup_{maize,arab}.csv` 等）。
水稻本来就走 `outputs/reproduce/author_rice_control_metrics_rr/`，未受影响。

迁移后已实测：`make_table2_data.py` 的 15 个输出文件 md5 **与迁移前逐字节一致**。

## 2026-10-15 删除说明

- 删除目标：`D:\pythonproject\knode\outputs\dynamicGP`（478 个文件，27.56 GB）
- 删除原因：除上面 2 个 CSV 外，其余（作者的 `Training_Validation_Results/Iteration_*/` 组件、
  `rrBLUP_benchmark_Training_Marker_Effects/Iteration_*/*.csv` 每个 ~415 MB 等）**没有任何活引用**。
- **服务器 `/data/fandata/dashuaige/outputs/dynamicGP`（33 GB）按用户要求保留未动**。

## 以后要重新生成这两个 CSV 的话

- 玉米：`reproduce/maize_magic/dynamicGP_MegaLMM_TP1_maize.R` 第 604 行读、第 617 行写，
  **两条路径都指向 `outputs/dynamicGP/...`**（本地已删，只能在服务器上跑；服务器那份目录还在）。
  跑完把 `.../Training_Validation_Results/rrBLUP_benchmarks_prediction_accuracies.csv`
  覆盖到本目录的 `_maize.csv` 即可。
- 拟南芥：原生成脚本是备份里的 `reproduce/_verify_bak_old_rr_20260927/dynamicgp_arabidopsis_all.R`
  （`base <- "outputs/dynamicGP/Arabidopsis/Training_Validation_Results"`），同样只能在服务器上跑。
