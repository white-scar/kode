# ==========================================
# 1. 安装和加载必要的包
# ==========================================
if (!require("data.table")) install.packages("data.table")
if (!require("rrBLUP")) install.packages("rrBLUP")

library(data.table)
library(rrBLUP)

# ==========================================
# 2. 读取原始基因型数据
# ==========================================
# 注意：Windows 路径中的括号要用双反斜杠或正斜杠
snp_file <- "D:/pythonproject/my_project/data/maize(500)/geno.numerical.csv"

cat("正在读取基因型数据...\n")
snp_dt <- fread(snp_file)

# 查看数据结构（调试用）
cat("数据维度：", dim(snp_dt), "\n")
cat("前几列名：", colnames(snp_dt)[1:5], "\n")

# 提取基因型 ID 作为行名（假设第一列列名可能是 "genotype" 或 "V1"）
# 这里直接取第一列的内容
genotype_ids <- snp_dt[[1]]  # 第一列的所有值

# 将数据转换为纯数值矩阵（剔除第一列）
cat("正在格式化 SNP 矩阵...\n")
snp_matrix <- as.matrix(snp_dt[, -1, with = FALSE])

# 检查唯一值（应该只有 0 和 2）
unique_vals <- unique(as.vector(snp_matrix))
cat("矩阵中的唯一值：", unique_vals, "\n")

# 将 0, 2 格式转换为 rrBLUP 需要的 -1, 1 格式
# 0 -> -1, 2 -> 1
snp_matrix_centered <- (snp_matrix / 2) * 2 - 1   # 等价于 (snp_matrix - 1)
# 或者直接用：
# snp_matrix_centered <- snp_matrix - 1

# 给矩阵赋予行名（非常重要，后续对齐全靠它）
rownames(snp_matrix_centered) <- genotype_ids

# 可选：检查是否有缺失值
if (any(is.na(snp_matrix_centered))) {
  cat("警告：存在缺失值，A.mat 会自动填补\n")
} else {
  cat("良好：无缺失值\n")
}

# ==========================================
# 3. 计算 GRM (Genomic Relationship Matrix)
# ==========================================
cat("正在使用 VanRaden 方法计算 GRM (可能需要一点时间，取决于 SNP 数量)...\n")

# A.mat 参数解释：
# - min.MAF: 最小等位基因频率，默认 0.05，可去掉稀有 SNP
# - max.missing: 最大缺失率，默认 0.5，这里无缺失不影响
grm_matrix <- A.mat(snp_matrix_centered, min.MAF = 0.05)

cat("GRM 计算完成！矩阵维度:", nrow(grm_matrix), "x", ncol(grm_matrix), "\n")

# ==========================================
# 4. 保存 GRM 文件供后续模型使用
# ==========================================
out_dir <- dirname(snp_file)

# 保存为 RDS 格式 (强烈推荐：R 语言读取极快，且不会丢失行名和列名)
rds_out <- file.path(out_dir, "geno_GRM.rds")
saveRDS(grm_matrix, rds_out)

# 同时保存一份 CSV 格式，方便用 Excel 或 Python 查看
grm_df <- as.data.frame(grm_matrix)
grm_df <- cbind(genotype = rownames(grm_df), grm_df)
csv_out <- file.path(out_dir, "geno_GRM.csv")
fwrite(grm_df, csv_out)

cat("🎉 GRM 已成功保存！\n")
cat("RDS 路径:", rds_out, "\n")
cat("CSV 路径:", csv_out, "\n")

# 可选：打印 GRM 的前几行和前几列，验证一下
cat("\nGRM 矩阵前5行前5列示例：\n")
print(grm_matrix[1:min(5, nrow(grm_matrix)), 1:min(5, ncol(grm_matrix))])