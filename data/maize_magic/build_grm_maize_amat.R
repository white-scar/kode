#!/usr/bin/env Rscript
# build_grm_maize_amat.R — 用拟南芥作者原版方法 (rrBLUP::A.mat, 0/1/2 直接喂) 计算玉米 GRM
#
# 目的: 对照"拟南芥 A.mat 做法" vs "玉米现有 VanRaden 公式做法"在玉米数据上的差异。
# 拟南芥方法 (reproduce_arabidopsis_all.R Part 1, 作者原码):
#   - 0/1/2 编码直接喂 A.mat (不做 -1 平移)
#   - MAF>=0.05 预过滤
#   - A.mat 默认 min.MAF (A.mat 内部 freq=(X+1)/2 -> p+0.5, 内部 MAF=min(p+0.5, 0.5-p))
# 玉米数据: MAGIC.SNP.70k.impute.maf.vcf.txt (552 系 x 70846 标记, 0/1/2)
# 流程: 全 552 算 A.mat -> 子集到 330 (与 build_grm.py Method A 同口径) -> 对比现有 GRM
#
# 用法: Rscript build_grm_maize_amat.R   (输出到 data/maize_magic/)

suppressMessages(library(rrBLUP))
suppressMessages(library(data.table))

setwd("d:/pythonproject/knode/data/maize_magic")

# ---------- 1. 加载 SNP (同 build_grm.py 口径: 空格分隔, 首列基因型 ID) ----------
# 表头行整行带引号("1:111303"), 数据行不带(1_89): fread 启发式会把带引号首行当表头直接丢弃,
# 即便 header=FALSE 也读不到它 -> 标记名单独用 readLines 从首行取, 数据用 header=FALSE 全量读.
snps_all <- data.table::fread("MAGIC.SNP.70k.impute.maf.vcf.txt", header = FALSE)
lines <- as.character(snps_all[[1]])                       # 全部 552 个基因型 ID
M <- as.matrix(snps_all[, -1, with = FALSE]); rm(snps_all); gc()   # 552 x 70846
marker_names <- gsub('"', '', unlist(strsplit(readLines("MAGIC.SNP.70k.impute.maf.vcf.txt", n = 1), "\\s+")))
if (length(marker_names) == ncol(M)) colnames(M) <- marker_names
rownames(M) <- lines
cat(sprintf("SNP matrix: %d genotypes x %d markers\n", nrow(M), ncol(M)))
cat(sprintf("first/last marker: %s ... %s\n", marker_names[1], marker_names[length(marker_names)]))
cat(sprintf("first genotype id: %s\n", lines[1]))

# ---------- 2. MAF>=0.05 预过滤 (与拟南芥 Part 1 一致) ----------
p <- colMeans(M) / 2
keep <- pmin(p, 1 - p) >= 0.05
M_qc <- M[, keep]; rm(M, p); gc()
cat(sprintf("After MAF>=0.05 filter: %d markers\n", ncol(M_qc)))

# ---------- 3. 拟南芥作者方法: 0/1/2 直接喂 A.mat (默认 min.MAF) ----------
G <- rrBLUP::A.mat(M_qc)
# A.mat 内部对 0/1/2 按 p+0.5 推 MAF, p>0.5 的标记会被内部丢弃 -> 统计实际用的标记数
p2 <- colMeans(M_qc) / 2
amin <- pmin(p2 + 0.5, 0.5 - p2)
cat(sprintf("A.mat-internal kept markers (default min.MAF=1/2n): %d of %d\n",
            sum(amin >= 1/(2*nrow(M_qc))), ncol(M_qc)))

cat(sprintf("Full 552 GRM diag [%.3f, %.3f] mean %.3f\n",
            min(diag(G)), max(diag(G)), mean(diag(G))))

# ---------- 4. 子集到 330 表型系 (同 build_grm.py Method A: 先全量再子集) ----------
pheno <- data.frame(bio_ID = data.table::fread("maize_trait_BLUP_50_NArmv_traits_afterClustering.csv")$bio_ID)
pheno_ids <- sort(unique(as.character(pheno$bio_ID)))
common <- sort(intersect(lines, pheno_ids))
cat(sprintf("Phenotype ids: %d, common: %d\n", length(pheno_ids), length(common)))

G_sub <- G[common, common]
g_out <- as.data.frame(G_sub); g_out <- cbind.data.frame(genotype = common, g_out)
data.table::fwrite(g_out, "GRM_330x330_amat_from552.csv")
cat(sprintf("Saved: GRM_330x330_amat_from552.csv  (%d x %d)\n", nrow(G_sub), ncol(G_sub)))

# ---------- 5. 对比现有 VanRaden 公式版 (GRM_330x330_methodA_from552.csv) ----------
old_dt <- data.table::fread("GRM_330x330_methodA_from552.csv")
# 首列是基因型 ID (字符), 删掉后其余全数值 -> 避免 as.matrix 全转字符
old_ids <- as.character(old_dt[[1]])
old_dt[[1]] <- NULL
old <- as.matrix(old_dt); storage.mode(old) <- "numeric"
rownames(old) <- old_ids; colnames(old) <- old_ids
# 按 common 顺序对齐行/列再比
old <- old[common, common]
A <- as.matrix(G_sub); B <- old
cat("\n=== 对比: A.mat(拟南芥法) vs VanRaden公式(build_grm.py) ===\n")
cat(sprintf("  dim        A.mat=%d VanRaden=%d\n", nrow(A), nrow(B)))
cat(sprintf("  corr       = %.6f\n", cor(as.vector(A), as.vector(B))))
cat(sprintf("  maxdiff    = %.6f\n", max(abs(A - B))))
cat(sprintf("  MAE        = %.6f\n", mean(abs(A - B))))
cat(sprintf("  diag mean  A.mat=%.4f  VanRaden=%.4f\n", mean(diag(A)), mean(diag(B))))
cat(sprintf("  offdiag mean A.mat=%.4f  VanRaden=%.4f\n",
            mean(A[lower.tri(A)]), mean(B[lower.tri(B)])))
# 线性缩放关系 (A.mat 内部按 p+0.5 推 MAF, 数值尺度可能与 VanRaden 差常数)
cat(sprintf("  ratio mean (A/V) = %.4f\n", mean(as.vector(A) / as.vector(B))))
cat("DONE\n")
