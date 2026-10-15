# dynamicGP_RR_innerens_arab.R
# DynamicGP-RR-BLUP (Arabidopsis) —— inner-5 集成版
#
# 与 reproduce/maize_magic/dynamicGP_MegaLMM_TP1_maize.R 的关系 (规则 8: 原样照抄模板):
#   * 数据加载 / 训练折 scaler / Schur-DMD (find_DMD_components, scale="none", r=2, time_gaps=0) /
#     A_r = phi %*% R %*% pinv(phi) / predict_dynamics / metrics / 输出列 全部与模板一致
#   * 只保留 DynamicGP-RR-BLUP 一条分支
#   * 改动仅一处: 把模板里 rr 的 "训练集内部 5 折 -> 选 best fold -> 预测验证"
#     改成 inner-5 集成, 并把训练集用满:
#        对每个 outer fold k, 训练集(folds1-5)按折标分成 5 个 inner fold;
#        第 j 个模型用其余 4/5 训练 (5 折轮转 => 训练集里每个基因型都被用到),
#        5 个模型分别预测 outer 验证集(fold 6) 的元素值, 取平均 -> R_ens, phi_ens;
#        再 A = phi_ens R_ens phi_ens^dagger -> rollout -> metrics.
#      (与 Kode 主方法的 inner-5 集成口径一致; 相比 best-fold 不再丢掉 1/5 训练基因型)
#
# 用法: Rscript dynamicGP_RR_innerens_arab.R [START_K] [END_K] [NCORES] [OUTDIR] [MAX_INNER]
#   默认 START_K=1, END_K=10  -> 一次跑满 10 个 fold-set (正式结果写 outputs/dynamicgp_rr_innerens/)
#   MAX_INNER: 只跑前几个 inner fold (冒烟用; 默认 5)
#   并行: 每个 fold-set 起一个进程 -> Rscript ... k k 8 outputs/dynamicgp_rr_innerens_par/fold$k 5
args <- commandArgs(trailingOnly = TRUE)
START_K   <- if(length(args)>=1) as.integer(args[1]) else 1
END_K     <- if(length(args)>=2) as.integer(args[2]) else 10      # 默认跑满 10 个 fold-set
NCORES    <- if(length(args)>=3) as.integer(args[3]) else 8
OUTDIR    <- if(length(args)>=4) args[4] else "outputs/dynamicgp_rr_innerens_arab"
MAX_INNER <- if(length(args)>=5) as.integer(args[5]) else 5

setwd("/data/fandata/dashuaige")
for (f in c("find_DMD_components.R","tensor_unroller.R","table_to_tensor.R",
            "A_r_from_phi_and_R.R","all_timepoints_predictor.R"))
  source(file.path("reproduce/R", f))
suppressPackageStartupMessages(library(rrBLUP))
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)
R_RANK <- 2
TIME_GAPS <- 0
Sys.setenv(OMP_NUM_THREADS = as.character(NCORES))

cat("Loading data ...\n")
# ---- Arabidopsis: 382 genotypes x 45 traits x 14 timepoints (DAS 8-21, no gaps -> time_gaps=0) ----
data <- read.csv("data/arabidopsis/BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv")[, -1]
cv   <- read.csv("data/arabidopsis/cross_validation_folds_w_validation_set_arabidopsis.csv")
gids <- unique(data$bio_ID)
snps_raw <- data.table::fread("data/arabidopsis/myGD_w_h.csv", data.table = FALSE)
snp_ids  <- snps_raw[[1]]
snps     <- data.matrix(snps_raw[, -1, drop = FALSE]); rownames(snps) <- snp_ids
snps     <- snps - 1                                  # GAPIT 0/1/2 -> -1/0/1
rm(snps_raw, snp_ids); gc()
snps <- as.matrix(snps[gids, ]); storage.mode(snps) <- "double"

# ---- rrBLUP: 用给定训练子集拟合 104 个元素(R 4 + phi 100), 预测 val_gids ----
rrblup_fit_predict <- function(train_mat, val_gids) {
  trn_gids <- rownames(train_mat); n_el <- ncol(train_mat)
  out <- matrix(NA, length(val_gids), n_el, dimnames = list(val_gids, colnames(train_mat)))
  Z_trn <- snps[trn_gids, , drop = FALSE]; Z_val <- snps[val_gids, , drop = FALSE]
  for (i in seq_len(n_el)) {
    y <- train_mat[, i]; names(y) <- trn_gids
    m <- rrBLUP::mixed.solve(y = y, Z = Z_trn)
    out[, i] <- as.numeric(Z_val %*% as.numeric(m$u)) + as.numeric(m$beta)
  }
  out
}

# ---- dynamicGP forward predictor + snapshot metrics (与模板一致) ----
predict_dynamics <- function(A_all, X_all) {
  lines <- intersect(dimnames(A_all)[[3]], dimnames(X_all)[[3]])
  traits <- dimnames(A_all)[[1]]; n_tp <- dim(X_all)[2]
  mat <- mat_rec <- array(NA, dim = c(length(traits), n_tp, length(lines)),
                          dimnames = list(traits, paste0("t", 1:n_tp), lines))
  for (ln in lines) {
    X <- X_all[,,ln]; A <- A_all[,,ln]
    m <- m_rec <- matrix(0, nrow(X), n_tp)
    m[,1] <- m_rec[,1] <- X[,1]
    for (t in 1:(n_tp-1)) {
      m[,t+1]     <- Re(A %*% X[,t])
      m_rec[,t+1] <- Re(A %*% m_rec[,t])
    }
    mat[,,ln] <- m; mat_rec[,,ln] <- m_rec
  }
  metrics <- function(P) {
    tr <- rep(traits, each = n_tp); tm <- rep(seq_len(n_tp), times = length(traits))
    PCC <- MSE <- NMSE <- R2 <- numeric(length(tr))
    for (i in seq_along(tr)) {
      trait <- tr[i]; t <- tm[i]
      y <- X_all[trait, t, ]; p <- P[trait, t, ]
      ok <- is.finite(y) & is.finite(p)
      if (sum(ok) < 3) { PCC[i] <- MSE[i] <- NMSE[i] <- R2[i] <- NA_real_; next }
      y <- y[ok]; p <- p[ok]
      mse <- mean((y-p)^2); vy <- var(y)
      MSE[i] <- mse; NMSE[i] <- if(vy>0) mse/vy else NA_real_
      R2[i] <- if(vy>0) 1-mse/vy else NA_real_; PCC[i] <- cor(y,p)
    }
    data.frame(Trait = tr, Time = paste0("t", tm), PCC = PCC, MSE = MSE, NMSE = NMSE, R2 = R2, stringsAsFactors = FALSE)
  }
  pooled <- function(P) {
    y <- as.vector(X_all[, -1, , drop = FALSE]); p <- as.vector(P[, -1, , drop = FALSE])
    ok <- is.finite(y) & is.finite(p); y <- y[ok]; p <- p[ok]
    mse <- mean((y-p)^2); vy <- var(y)
    c(PCC = cor(y,p), MSE = mse, NMSE = if(vy>0) mse/vy else NA_real_, R2 = if(vy>0) 1-mse/vy else NA_real_)
  }
  list(iter = metrics(mat), rec = metrics(mat_rec), pool_iter = pooled(mat), pool_rec = pooled(mat_rec))
}

# ---- one CV iteration: rrBLUP inner-5 集成 ----
run_iteration <- function(k) {
  cvcol <- cv[, c(1, k+1)]; rownames(cvcol) <- cvcol[, 1]
  folds <- cvcol[gids, 2]
  trn_idx <- which(folds != 6); val_idx <- which(folds == 6)
  trn_gids <- gids[trn_idx]; val_gids <- gids[val_idx]
  cat(sprintf("\n=== iteration %d : train=%d val=%d ===\n", k, length(trn_idx), length(val_idx)))

  # 训练折 scaler: min/max 只在训练子集上拟合 (与模板一致)
  trn_dat <- data[data$bio_ID %in% trn_gids, ]
  val_dat <- data[data$bio_ID %in% val_gids, ]
  trn_min <- sapply(trn_dat[, -c(1:3)], min, na.rm = TRUE)
  trn_max <- sapply(trn_dat[, -c(1:3)], max, na.rm = TRUE)
  norm_tr <- function(df, mn, mx) {
    df[, -c(1:3)] <- as.data.frame(mapply(function(x, a, b) (x - a) / (b - a),
                                          df[, -c(1:3)], mn, mx, SIMPLIFY = FALSE))
    df
  }
  trn_dat <- norm_tr(trn_dat, trn_min, trn_max)
  val_dat <- norm_tr(val_dat, trn_min, trn_max)
  cat("  Schur-DMD: training subset ...\n")
  trn_comp <- find_DMD_components(trn_dat, r = R_RANK, method = "schur.DMD", scale = "none", time_gaps = TIME_GAPS)
  trn_R   <- as.matrix(tensor_unroller(trn_comp$R_all))     # train x 4
  trn_phi <- as.matrix(tensor_unroller(trn_comp$phi_all))   # train x 100
  trn_flat <- cbind(trn_R, trn_phi)                         # train x 104
  cat("  Schur-DMD: validation subset ...\n")
  val_comp <- find_DMD_components(val_dat, r = R_RANK, method = "schur.DMD", scale = "none", time_gaps = TIME_GAPS)
  val_X   <- val_comp$X_all
  val_R_true <- as.matrix(tensor_unroller(val_comp$R_all))
  val_phi_true <- as.matrix(tensor_unroller(val_comp$phi_all))

  # ---- inner-5 集成: 第 j 个模型用训练集的 4/5, 预测 outer 验证集 ----
  inner <- cvcol[trn_gids, 2]   # 折标(1-5); 与模板一致: cvcol = cv[,c(1,k+1)], rownames = gid
  n_inner <- min(5L, MAX_INNER)
  VR <- list(); VP <- list()
  for (j in 1:5) {
    itr <- which(inner != j); iva <- trn_gids[inner == j]
    if (j > n_inner) { cat(sprintf("  [rrBLUP] inner %d: skipped (MAX_INNER=%d)\n", j, n_inner)); next }
    cat(sprintf("  [rrBLUP] inner %d: train=%d (holdout %d) ...\n", j, length(itr), length(iva)))
    pr <- rrblup_fit_predict(trn_flat[itr, , drop = FALSE], val_gids)
    VR[[j]] <- pr[, seq_len(ncol(trn_R)),  drop = FALSE]
    VP[[j]] <- pr[, ncol(trn_R) + seq_len(ncol(trn_phi)), drop = FALSE]
  }
  R_rr   <- Reduce(`+`, VR) / length(VR)
  phi_rr <- Reduce(`+`, VP) / length(VP)
  rownames(R_rr) <- val_gids; rownames(phi_rr) <- val_gids
  cat(sprintf("  [rrBLUP] inner-5 集成完成 (用了 %d 个模型)\n", length(VR)))

  comp_acc <- function(pred, truth) {
    common <- intersect(rownames(pred), rownames(truth))
    mean(sapply(seq_len(ncol(pred)), function(j) cor(truth[common, j], pred[common, j], use = "complete.obs")), na.rm = TRUE)
  }
  cat(sprintf("  [diag] R comp acc: rr(innerens)=%.3f | phi comp acc: rr(innerens)=%.3f\n",
              comp_acc(R_rr, val_R_true), comp_acc(phi_rr, val_phi_true)))

  evaluate <- function(Rp, PHIp) {
    Rv   <- table_to_tensor(as.data.frame(Rp),   component = "R",   r = R_RANK)
    PHIv <- table_to_tensor(as.data.frame(PHIp), component = "phi", r = R_RANK)
    Av   <- A_r_from_phi_and_R(PHIv, Rv)
    m    <- predict_dynamics(Av, val_X)
    list(iter = as.data.frame(m$iter), rec = as.data.frame(m$rec),
         pool_iter = m$pool_iter, pool_rec = m$pool_rec)
  }
  cat("  evaluate rrBLUP (inner-5 集成) ...\n")
  e_rr <- evaluate(R_rr, phi_rr)
  list(rr = e_rr)
}

# ---- main ----
store <- list()
for (k in START_K:END_K) {
  r <- run_iteration(k)
  store[[as.character(k)]] <- r
  saveRDS(r, file.path(OUTDIR, sprintf("results_iter_%02d.rds", k)))
}

# ---- 汇总写表 (与模板同样的输出列) ----
if (length(store) > 0) {
  confs <- c("rr_rec" = "rec", "rr_iter" = "iter")
  tab <- NULL
  for (meth in names(confs)) {
    cfg <- confs[[meth]]
    for (k in names(store)) {
      df <- store[[k]]$rr[[cfg]]
      df$Method <- meth; df$Iteration <- as.integer(k)
      tab <- rbind(tab, df)
    }
  }
  tab <- tab[, c("Method","Iteration","Trait","Time","PCC","MSE","NMSE","R2")]
  write.csv(tab, file.path(OUTDIR, "snapshot_metrics_long.csv"), row.names = FALSE)
  agg <- aggregate(cbind(PCC, MSE, NMSE, R2) ~ Method + Time, data = tab, FUN = mean, na.rm = TRUE)
  write.csv(agg, file.path(OUTDIR, "snapshot_metrics_by_time.csv"), row.names = FALSE)
  tab_not1 <- tab[tab$Time != "t1", ]
  ov <- aggregate(cbind(PCC, MSE, NMSE, R2) ~ Method, data = tab_not1, FUN = mean, na.rm = TRUE)
  write.csv(ov, file.path(OUTDIR, "snapshot_metrics_overall.csv"), row.names = FALSE)
  cat("\n=== snapshot_metrics_overall.csv (不含 t1) ===\n"); print(ov)
  cat("\nrows:", nrow(tab), " -> ", file.path(OUTDIR, "snapshot_metrics_long.csv"), "\n")
}
cat("ALL DONE\n")