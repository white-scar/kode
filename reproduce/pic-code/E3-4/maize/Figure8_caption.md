**补充图 4 | 玉米 MAGIC 群体中两种方法相对 rrBLUP 基准的预测增益随可预测性阈值的变化。**
横轴为可预测性阈值（按各方法自身逐性状平均 PCC 筛选性状，步长 0.05，DAS 16-47 上平均），纵轴为入选性状相对 rrBLUP 基准的平均增益；每档左柱为迭代协议、右柱为递归协议，段色蓝＝DynamicGP+MegaLMM+TP1、红＝Kode+TP1，柱内白底数字为该阈值下的性状数，柱顶数字为该阈值下的最大增益。基准为作者流程中的 rrBLUP 臂（逐性状、逐时间点的 rrBLUP 预测）。增益随阈值上升并趋于平台：迭代协议下 Kode+TP1 最高 0.729（阈值 0.65，49 性状）高于 DynamicGP+MegaLMM+TP1 的 0.509；递归协议下 0.409 对 0.389。

**Extended Data Fig. 4 | Accuracy gain over the rrBLUP baseline as a function of the predictability threshold (maize MAGIC).**
x, threshold on each method's own per-trait mean PCC (step 0.05, averaged over DAS 16-47); y, mean gain of the selected traits over the rrBLUP baseline (left bar, iterative; right bar, recursive; blue, DynamicGP+MegaLMM+TP1; red, Kode+TP1); in-bar numbers are the number of traits passing the threshold, top numbers the largest gain at that threshold. Max gain: Kode+TP1 0.729 (threshold 0.65, 49 traits) vs DynamicGP+MegaLMM+TP1 0.509 (iterative); 0.409 vs 0.389 (recursive).
