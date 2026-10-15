
# ============================================================================
# dynamicGP-MegaLMM and dynamicGP-MegaLMM+TP1 (maize) reproduction
# Protocol B: train/validation are min-max scaled SEPARATELY (per subset),
# matching dynamicGP_maize.R (no leakage of the validation set).
#
# Metrics (snapshot, per trait x timepoint across validation genotypes):
#   PCC, MSE, NMSE (= MSE / var(y_true)), R2 (= 1 - NMSE)
#
# Methods compared (same DMD components, different core GP):
#   1. dynamicGP-MegaLMM      : MegaLMM (GRM) predicts R and the two phi vectors
#   2. dynamicGP-MegaLMM+TP1  : same + all 50 t=1 traits as secondary traits
#   3. dynamicGP-RR-BLUP      : rrBLUP (SNP markers) predicts R and phi (baseline)
# ============================================================================
options(stringsAsFactors = FALSE)
suppressPackageStartupMessages(library(MegaLMM))
suppressPackageStartupMessages(library(rrBLUP))

args <- commandArgs(trailingOnly = TRUE)
START_K <- if(length(args)>=1) as.integer(args[1]) else 1
END_K   <- if(length(args)>=2) as.integer(args[2]) else 1
BURN    <- if(length(args)>=3) as.integer(args[3]) else 200
N_ITER  <- if(length(args)>=4) as.integer(args[4]) else 300
THIN    <- if(length(args)>=5) as.integer(args[5]) else 2
NCORES  <- if(length(args)>=6) as.integer(args[6]) else 12
OUTDIR  <- if(length(args)>=7) args[7] else "outputs/reproduce/dynamicGP_MegaLMM_maize"

setwd("/data/fandata/dashuaige")
for (f in c("find_DMD_components.R","tensor_unroller.R","table_to_tensor.R",
            "A_r_from_phi_and_R.R","all_timepoints_predictor.R"))
  source(file.path("reproduce/R", f))

dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)
R_RANK <- 2
N_GENO <- 330

# limit CPU (MegaLMM is CPU-only MCMC; keep footprint modest on shared server)
Sys.setenv(OMP_NUM_THREADS = as.character(NCORES))
try(suppressWarnings(MegaLMM::set_MegaLMM_nthreads(NCORES)), silent = TRUE)
try(suppressWarnings(MegaLMM::set_omp_nthreads(NCORES)), silent = TRUE)

# ---- Fix MegaLMM 0.9.5 bug: Eta/Eta_mean posterior arrays uninitialized ----
fix_megalmm_posterior <- function(st) {
  for (p in st$Posterior$posteriorSample_params) {
    if (is.null(st$Posterior[[p]]) && !is.null(st$current_state[[p]]) && length(dim(st$current_state[[p]])) == 2) {
      st$Posterior[[p]] <- array(0, dim = c(0, dim(st$current_state[[p]])))
      dimnames(st$Posterior[[p]])[2:3] <- dimnames(st$current_state[[p]])
    }
  }
  for (p in st$Posterior$posteriorMean_params) {
    if (is.null(st$Posterior[[p]]) && !is.null(st$current_state[[p]]) && length(dim(st$current_state[[p]])) == 2) {
      st$Posterior[[p]] <- array(0, dim = dim(st$current_state[[p]]))
      dimnames(st$Posterior[[p]]) <- dimnames(st$current_state[[p]])
    }
  }
  st
}

cat("Loading data ...\n")
data <- read.csv("data/maize_magic/maize_trait_BLUP_50_NArmv_traits_afterClustering.csv")[, -1]
cv   <- read.csv("data/maize_magic/cross_validation_folds_w_validation_set_maize.csv")
grm  <- as.matrix(read.csv("data/maize_magic/GRM_330x330_methodA_from552.csv", row.names = 1, check.names = FALSE))
snps <- readRDS("data/maize_magic/snps.rds")

gids <- unique(data$bio_ID)
grm  <- grm[gids, gids]
K    <- as.matrix(Matrix::forceSymmetric(grm))
colnames(K) <- rownames(K) <- gids
snps <- as.matrix(snps[gids, ]); storage.mode(snps) <- "double"
lmm_data <- data.frame(Line = gids, row.names = gids)
trait_cols <- colnames(data)[-c(1,2,3)]

# t=1 (DAS 15) secondary traits, min-max normalized across the 330 lines (observed -> no leakage)
tp1_raw <- data[data$DAS == 15, ]
rownames(tp1_raw) <- tp1_raw$bio_ID
tp1 <- as.data.frame(lapply(tp1_raw[, trait_cols], function(x) (x - min(x))/(max(x) - min(x))))
rownames(tp1) <- tp1_raw$bio_ID
tp1 <- as.matrix(tp1[gids, , drop = FALSE])
storage.mode(tp1) <- "double"
cat(sprintf("n genotypes=%d, n traits=%d, TP1 dim %d x %d\n", length(gids), length(trait_cols), nrow(tp1), ncol(tp1)))

# ---- MegaLMM fit/predict: focal Y (train rows observed, val rows NA) [+ optional secondary] ----
megalmm_predict <- function(train_mat, val_gids, secondary = NULL, k_factors, run_id) {
  # train_mat : matrix (rownames = training genotype ids)
  # secondary : matrix (rownames = all gids), observed for everyone
  focal <- matrix(NA, length(gids), ncol(train_mat), dimnames = list(gids, colnames(train_mat)))
  focal[rownames(train_mat), ] <- as.matrix(train_mat)
  Y <- focal
  if (!is.null(secondary)) Y <- cbind(focal, secondary[gids, , drop = FALSE])
  Y2 <- Y
  Y2[val_gids, seq_len(ncol(focal))] <- NA   # mask focal for validation genotypes only

  rp <- MegaLMM_control()
  rp$K <- k_factors; rp$h2_divisions <- 20L; rp$burn <- 0L; rp$thin <- THIN
  st <- setup_model_MegaLMM(Y = Y2, formula = ~ (1|Line), data = lmm_data,
                            relmat = list(Line = K), run_parameters = rp, run_ID = run_id)
  st <- set_priors_MegaLMM(st, MegaLMM_priors(
    tot_Y_var = list(V = 0.5, nu = 5), tot_F_var = list(V = 18/20, nu = 20),
    h2_priors_resids_fun = function(h2s,n) 1, h2_priors_factors_fun = function(h2s,n) 1,
    Lambda_prior = list(sampler = sample_Lambda_prec_ARD, Lambda_df = 3,
                        delta_1 = list(shape = 2, rate = 1), delta_2 = list(shape = 3, rate = 1),
                        delta_iterations_factor = 100)))
  st <- initialize_variables_MegaLMM(st)
  st <- fix_megalmm_posterior(st)
  maps <- make_Missing_data_map(st, max_NA_groups = Inf, verbose = FALSE)
  st <- set_Missing_data_map(st, maps$Missing_data_map)
  st <- initialize_MegaLMM(st, ncores = NCORES)
  for (b in 1:2) {
    st <- sample_MegaLMM(st, BURN)
    try(suppressWarnings(st <- reorder_factors(st, drop_cor_threshold = 0.6)), silent = TRUE)
    st <- clear_Posterior(st)
    st <- fix_megalmm_posterior(st)
  }
  st <- sample_MegaLMM(st, N_ITER)
  st <- save_posterior_chunk(st)
  Eta <- load_posterior_param(st, "Eta_mean")
  Eta[val_gids, seq_len(ncol(focal)), drop = FALSE]
}

# ---- rrBLUP predict (SNP markers): train on train_mat, predict validation ----
# Author protocol (dynamicGP_maize.R): internal 5-fold CV on R+phi COMBINED -> best_fold -> predict validation
rrblup_predict <- function(train_mats, val_gids, cvcol) {
  trn_gids <- rownames(train_mats[[1]])
  all_Y <- do.call(cbind, train_mats)
  n_el <- ncol(all_Y)
  fold_acc <- matrix(NA, n_el, 5)
  store <- vector("list", n_el)
  for (i in seq_len(n_el)) {
    store[[i]] <- vector("list", 5)
    y <- all_Y[, i]; names(y) <- trn_gids
    for (j in 1:5) {
      tr <- trn_gids[which(cvcol[trn_gids, 2] != j)]
      te <- trn_gids[which(cvcol[trn_gids, 2] == j)]
      m <- rrBLUP::mixed.solve(y = y[tr], Z = snps[tr, , drop = FALSE])
      me <- as.numeric(m$u); b <- as.numeric(m$beta)
      pred_te <- as.numeric(snps[te, , drop = FALSE] %*% me) + b
      fold_acc[i, j] <- cor(pred_te, y[te], use = "complete.obs")
      store[[i]][[j]] <- list(BLUE = b, me = me)
    }
  }
  best_fold <- which.max(colMeans(fold_acc, na.rm = TRUE))
  out <- matrix(NA, length(val_gids), n_el, dimnames = list(val_gids, colnames(all_Y)))
  for (i in seq_len(n_el)) {
    ss <- store[[i]][[best_fold]]
    out[, i] <- as.numeric(snps[val_gids, , drop = FALSE] %*% ss$me) + ss$BLUE
  }
  out
}

# ---- dynamicGP forward predictor + snapshot metrics ----
predict_dynamics <- function(A_all, X_all) {
  lines <- intersect(dimnames(A_all)[[3]], dimnames(X_all)[[3]])
  traits <- dimnames(A_all)[[1]]
  n_tp <- dim(X_all)[2]
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
    tr <- rep(traits, each = n_tp)
    tm <- rep(seq_len(n_tp), times = length(traits))
    PCC <- MSE <- NMSE <- R2 <- numeric(length(tr))
    for (i in seq_along(tr)) {
      trait <- tr[i]; t <- tm[i]
      y <- X_all[trait, t, ]; p <- P[trait, t, ]
      ok <- is.finite(y) & is.finite(p)
      if (sum(ok) < 3) { PCC[i] <- MSE[i] <- NMSE[i] <- R2[i] <- NA_real_; next }
      y <- y[ok]; p <- p[ok]
      mse <- mean((y-p)^2); vy <- var(y)
      MSE[i] <- mse
      NMSE[i] <- if(vy>0) mse/vy else NA_real_
      R2[i]   <- if(vy>0) 1-mse/vy else NA_real_
      PCC[i]  <- cor(y,p)
    }
    data.frame(Trait = tr, Time = paste0("t", tm), PCC = PCC, MSE = MSE, NMSE = NMSE, R2 = R2, stringsAsFactors = FALSE)
  }
  pooled <- function(P) {
    y <- as.vector(X_all[, -1, , drop = FALSE]); p <- as.vector(P[, -1, , drop = FALSE])
    ok <- is.finite(y) & is.finite(p)
    y <- y[ok]; p <- p[ok]
    mse <- mean((y-p)^2); vy <- var(y)
    c(PCC = cor(y,p), MSE = mse, NMSE = if(vy>0) mse/vy else NA_real_, R2 = if(vy>0) 1-mse/vy else NA_real_)
  }
  list(iter = metrics(mat), rec = metrics(mat_rec),
       pool_iter = pooled(mat), pool_rec = pooled(mat_rec))
}

# ---- one CV iteration (Protocol B) ----

build_val_X <- function(d, gids, traits) {
  d <- d[order(d$DAS), ]
  d[, traits] <- as.data.frame(lapply(d[, traits, drop = FALSE], function(x) {
    rng <- diff(range(x, na.rm = TRUE)); if (rng > 1e-12) (x - min(x, na.rm = TRUE)) / rng else rep(0, length(x))
  }))
  times <- sort(unique(d$DAS))
  X_all <- array(NA, dim = c(length(traits), length(times), length(gids)),
                 dimnames = list(traits, paste0("t", times), gids))
  for (i in seq_along(gids)) {
    X_all[, , i] <- t(as.matrix(d[d$bio_ID == gids[i], traits, drop = FALSE]))
  }
  X_all
}

run_iteration <- function(k) {
  cvcol <- cv[, c(1, k+1)]; rownames(cvcol) <- cvcol[, 1]
  folds <- cvcol[gids, 2]
  trn_idx <- which(folds != 6); val_idx <- which(folds == 6)
  trn_gids <- gids[trn_idx]; val_gids <- gids[val_idx]
  cat(sprintf("\n=== iteration %d : train=%d val=%d ===\n", k, length(trn_idx), length(val_idx)))

  # DMD components computed SEPARATELY on training and validation subsets
  cat("  Schur-DMD: training subset ...\n")
  trn_comp <- find_DMD_components(data[data$bio_ID %in% trn_gids & data$DAS <= 40, ], r = R_RANK, method = "schur.DMD", time_gaps = 3)
  trn_R   <- as.matrix(tensor_unroller(trn_comp$R_all))     # train x 4
  trn_phi <- as.matrix(tensor_unroller(trn_comp$phi_all))   # train x 100
  cat("  Schur-DMD: validation subset ...\n")
  val_X <- build_val_X(data[data$bio_ID %in% val_gids & data$DAS >= 40, ], val_gids, trait_cols)
  val_R_true <- matrix(0, length(val_gids), 4, dimnames = list(val_gids, NULL))
  val_phi_true <- matrix(0, length(val_gids), 100, dimnames = list(val_gids, NULL))
  sv1_cols <- grep("SV1$", colnames(trn_phi)); sv2_cols <- grep("SV2$", colnames(trn_phi))

  kR <- function(nf) min(floor(length(gids)/4), floor(nf/2))
  run_id <- function(m, k) sprintf("%s/run_%02d_%s", OUTDIR, k, m)

  # -------- dynamicGP-MegaLMM --------
  cat("  [MegaLMM] R (k=%d) ...\n")
  R_mega   <- megalmm_predict(trn_R, val_gids, k_factors = kR(4), run_id = run_id("mega_R", k))
  cat("  [MegaLMM] phi SV1 (k=%d) ...\n")
  p1_mega  <- megalmm_predict(trn_phi[, sv1_cols, drop = FALSE], val_gids, k_factors = kR(50), run_id = run_id("mega_p1", k))
  cat("  [MegaLMM] phi SV2 (k=%d) ...\n")
  p2_mega  <- megalmm_predict(trn_phi[, sv2_cols, drop = FALSE], val_gids, k_factors = kR(50), run_id = run_id("mega_p2", k))
  phi_mega <- matrix(NA, length(val_gids), ncol(trn_phi), dimnames = list(val_gids, colnames(trn_phi)))
  phi_mega[, sv1_cols] <- p1_mega; phi_mega[, sv2_cols] <- p2_mega

  # -------- dynamicGP-MegaLMM+TP1 --------
  cat("  [MegaLMM+TP1] R (k=%d) ...\n")
  R_tp1   <- megalmm_predict(trn_R, val_gids, secondary = tp1, k_factors = kR(4+50), run_id = run_id("tp1_R", k))
  cat("  [MegaLMM+TP1] phi SV1 (k=%d) ...\n")
  p1_tp1  <- megalmm_predict(trn_phi[, sv1_cols, drop = FALSE], val_gids, secondary = tp1, k_factors = kR(50+50), run_id = run_id("tp1_p1", k))
  cat("  [MegaLMM+TP1] phi SV2 (k=%d) ...\n")
  p2_tp1  <- megalmm_predict(trn_phi[, sv2_cols, drop = FALSE], val_gids, secondary = tp1, k_factors = kR(50+50), run_id = run_id("tp1_p2", k))
  phi_tp1 <- matrix(NA, length(val_gids), ncol(trn_phi), dimnames = list(val_gids, colnames(trn_phi)))
  phi_tp1[, sv1_cols] <- p1_tp1; phi_tp1[, sv2_cols] <- p2_tp1

  # -------- dynamicGP-RR-BLUP --------
  cat("  [rrBLUP] R + phi ...\n")
  pred_rr <- rrblup_predict(list(R = trn_R, phi = trn_phi), val_gids, cvcol)
  R_rr   <- pred_rr[, seq_len(ncol(trn_R)), drop = FALSE]
  phi_rr <- pred_rr[, ncol(trn_R) + seq_len(ncol(trn_phi)), drop = FALSE]

  comp_acc <- function(pred, truth) {
    common <- intersect(rownames(pred), rownames(truth))
    mean(sapply(seq_len(ncol(pred)), function(j) cor(truth[common, j], pred[common, j], use = "complete.obs")), na.rm = TRUE)
  }
  cat(sprintf("  [diag] R comp acc: rr=%.3f mega=%.3f tp1=%.3f\n",
              comp_acc(R_rr, val_R_true), comp_acc(R_mega, val_R_true), comp_acc(R_tp1, val_R_true)))
  cat(sprintf("  [diag] phi comp acc: rr=%.3f mega=%.3f tp1=%.3f\n",
              comp_acc(phi_rr, val_phi_true), comp_acc(phi_mega, val_phi_true), comp_acc(phi_tp1, val_phi_true)))

  # -------- reconstruct Ar and predict dynamics --------
  evaluate <- function(Rp, PHIp, tag) {
    Rv   <- table_to_tensor(as.data.frame(Rp),   component = "R",   r = R_RANK)
    PHIv <- table_to_tensor(as.data.frame(PHIp), component = "phi", r = R_RANK)
    Av   <- A_r_from_phi_and_R(PHIv, Rv)
    m    <- predict_dynamics(Av, val_X)
    list(iter = as.data.frame(m$iter), rec = as.data.frame(m$rec), pool_iter = m$pool_iter, pool_rec = m$pool_rec)
  }
  cat("  evaluate MegaLMM ...\n");   e_mega <- evaluate(R_mega, phi_mega, "mega")
  cat("  evaluate MegaLMM+TP1 ...\n"); e_tp1 <- evaluate(R_tp1, phi_tp1, "tp1")
  cat("  evaluate rrBLUP ...\n");     e_rr   <- evaluate(R_rr, phi_rr, "rr")

  list(mega = e_mega, tp1 = e_tp1, rr = e_rr)
}

# ---- main ----
store <- list()
for (k in START_K:END_K) {
  r <- run_iteration(k)
  store[[as.character(k)]] <- r
  saveRDS(r, file.path(OUTDIR, sprintf("results_iter_%02d.rds", k)))
}

long <- list()
for (k in names(store)) {
  r <- store[[k]]
  for (meth in names(r)) for (cfg in c("iter","rec")) {
    df <- r[[meth]][[cfg]]
    df$Method <- paste0(meth, "_", cfg)
    df$Iteration <- k
    long[[paste(k,meth,cfg)]] <- df
  }
}
tab <- do.call(rbind, long)
tab <- tab[, c("Method","Iteration","Trait","Time","PCC","MSE","NMSE","R2")]
write.csv(tab, file.path(OUTDIR, "snapshot_metrics_long.csv"), row.names = FALSE)

agg <- aggregate(cbind(PCC, MSE, NMSE, R2) ~ Method + Time, data = tab, FUN = mean, na.rm = TRUE)
write.csv(agg, file.path(OUTDIR, "snapshot_metrics_by_time.csv"), row.names = FALSE)
tab_not1 <- tab[tab$Time != "t1", ]
ov <- aggregate(cbind(PCC, MSE, NMSE, R2) ~ Method, data = tab_not1, FUN = mean, na.rm = TRUE)
write.csv(ov, file.path(OUTDIR, "snapshot_metrics_overall.csv"), row.names = FALSE)

pool_rows <- list()
for (k in names(store)) {
  r <- store[[k]]
  for (meth in names(r)) {
    for (cfg in c("iter","rec")) {
      v <- r[[meth]][[paste0("pool_", cfg)]]
      pool_rows[[paste(k,meth,cfg)]] <- data.frame(Method=paste0(meth,"_",cfg), Iteration=k,
        PCC=v["PCC"], MSE=v["MSE"], NMSE=v["NMSE"], R2=v["R2"], row.names=NULL)
    }
  }
}
pool <- do.call(rbind, pool_rows)
write.csv(pool, file.path(OUTDIR, "snapshot_metrics_pooled.csv"), row.names = FALSE)

cat("\n===== overall snapshot metrics (mean over traits & timepoints) =====\n")
print(ov, row.names = FALSE)
cat("\n===== pooled snapshot metrics (all validation predictions pooled) =====\n")
print(pool, row.names = FALSE)
cat("\nDONE\n")
