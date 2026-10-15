# -*- coding: utf-8 -*-
# _author_tp1_export_rice.R -- 作者 fig2 链路第 1 段: 导出每折的 DMD 分量 (rice, 连续段 DAS30-42)
#   口径: 训练折(folds1-5) per-trait min/max 做 scaler, DMD 用 scale="none" 且 time_gaps=0 (水稻连续, 无缺口)
#   输出: <OUTDIR>/components/iter%02d_{trn_R,trn_phi,val_R_true,val_phi_true}.csv
#   用法: Rscript _author_tp1_export_rice.R [K0] [K1] [OUTDIR]

args <- commandArgs(trailingOnly=TRUE)
K0 <- if(length(args)>=1) as.integer(args[1]) else 1
K1 <- if(length(args)>=2) as.integer(args[2]) else 10
OUTDIR <- if(length(args)>=3) args[3] else "outputs/reproduce/author_tp1_rice"
ROOT <- local({
  fa <- grep("^--file=", commandArgs(FALSE), value=TRUE)
  start <- if (length(fa)) dirname(normalizePath(sub("^--file=", "", fa[1]))) else getwd()
  p <- start; for (i in 1:8) { if (dir.exists(file.path(p,"data")) && dir.exists(file.path(p,"outputs"))) break; p <- dirname(p) }
  p
})
setwd(ROOT)
for (f in c("find_DMD_components.R","tensor_unroller.R","table_to_tensor.R")) source(file.path("reproduce/R", f))
COMP <- file.path(OUTDIR, "components"); dir.create(COMP, recursive=TRUE, showWarnings=FALSE)
R_RANK <- 2

data <- read.csv("data/rice_salinity/rice_indica_control_traits.csv")[, -1]
data <- data[data$DAS >= 30, ]                      # 只保留连续段 DAS 30-42 (丢掉 DAS28)
data$bio_ID <- as.character(data$bio_ID)
cv <- read.csv("data/rice_salinity/rice_cv_10fold.csv", row.names=1); rownames(cv) <- as.character(rownames(cv))
gids <- unique(data$bio_ID); trait_cols <- colnames(data)[-c(1,2,3)]
cat(sprintf("rice: N=%d T=%d P=%d DAS=%s\n", length(gids), length(unique(data$DAS)), length(trait_cols),
            paste(sort(unique(data$DAS)), collapse=",")))

for (k in K0:K1) {
  fc <- cv[gids, k]; trn <- gids[fc != 6]; val <- gids[fc == 6]
  mn <- sapply(data[data$bio_ID %in% trn, trait_cols], min, na.rm=TRUE)
  mx <- sapply(data[data$bio_ID %in% trn, trait_cols], max, na.rm=TRUE)
  rg <- pmax(mx - mn, 1e-8)
  dn <- data
  dn[, trait_cols] <- sweep(sweep(as.matrix(dn[, trait_cols]), 2, mn, "-"), 2, rg, "/")
  c1 <- find_DMD_components(dn[dn$bio_ID %in% trn, ], r = R_RANK, method = "schur.DMD", scale = "none", time_gaps = 0)
  c2 <- find_DMD_components(dn[dn$bio_ID %in% val, ], r = R_RANK, method = "schur.DMD", scale = "none", time_gaps = 0)
  R1 <- as.matrix(tensor_unroller(c1$R_all));   P1 <- as.matrix(tensor_unroller(c1$phi_all))
  R2 <- as.matrix(tensor_unroller(c2$R_all));   P2 <- as.matrix(tensor_unroller(c2$phi_all))
  write.csv(R1, file.path(COMP, sprintf("iter%02d_trn_R.csv", k)))
  write.csv(P1, file.path(COMP, sprintf("iter%02d_trn_phi.csv", k)))
  write.csv(R2, file.path(COMP, sprintf("iter%02d_val_R_true.csv", k)))
  write.csv(P2, file.path(COMP, sprintf("iter%02d_val_phi_true.csv", k)))
  cat(sprintf("  iter %d: train %d x (R %d, phi %d) | val %d x (R %d, phi %d)\n",
              k, nrow(R1), ncol(R1), ncol(P1), nrow(R2), ncol(R2), ncol(P2)))
}
cat("EXPORT DONE\n")