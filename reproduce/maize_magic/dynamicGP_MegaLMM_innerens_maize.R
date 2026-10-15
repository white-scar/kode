# ============================================================================
# dynamicGP-MegaLMM (maize MAGIC)  --  inner-5 ensemble version
# ============================================================================
# Split out of reproduce/maize_magic/dynamicGP_MegaLMM_TP1_maize.R (rule 8:
# copy the template line-by-line).  This file keeps ONLY the
# dynamicGP-MegaLMM branch (GRM only, NO TP1 secondary traits).
#
# Copied unchanged from the template:
#   data loading / training-fold scaler (min/max fitted on folds1-5 only) /
#   Schur-DMD (find_DMD_components, method="schur.DMD", scale="none", r=2) /
#   A_r = phi %*% R %*% pinv(phi) / predict_dynamics / metrics /
#   output file names and column names.
#
# Single change vs. the template: the template ran ONE MegaLMM fit on the
# whole training set (folds1-5) and then predicted the validation set
# (fold6).  Here we use inner-5 ensemble instead, i.e. for each outer fold k:
#   inner <- fold label (1..5) of the training gids
#   for j in 1..5:
#     train_mat_j = training gids with inner != j   (4/5 of the training set)
#     fit MegaLMM on train_mat_j, predict the outer validation gids (fold6)
#   average the 5 R vectors and the 5 phi vectors  -> R_ens, phi_ens
#   then A = phi_ens R_ens phi_ens^dagger -> rollout -> metrics
# (same ensemble protocol as the Kode main method)
#
# Usage:
#   Rscript dynamicGP_MegaLMM_innerens_maize.R [START_K] [END_K] [BURN] [N_ITER] [THIN] [NCORES] [OUTDIR] [MAX_INNER]
#   defaults: START_K=1 END_K=1 BURN=200 N_ITER=300 THIN=2 NCORES=12
#             OUTDIR=outputs/dynamicgp_megalmm_innerens MAX_INNER=5
#   MAX_INNER: only run the first few inner folds (smoke test); default 5.
# ============================================================================
options(stringsAsFactors = FALSE)
suppressPackageStartupMessages(library(MegaLMM))
suppressPackageStartupMessages(library(Matrix))

args <- commandArgs(trailingOnly = TRUE)
START_K   <- if(length(args)>=1) as.integer(args[1]) else 1
END_K     <- if(length(args)>=2) as.integer(args[2]) else 1
BURN      <- if(length(args)>=3) as.integer(args[3]) else 200
N_ITER    <- if(length(args)>=4) as.integer(args[4]) else 300
THIN      <- if(length(args)>=5) as.integer(args[5]) else 2
NCORES    <- if(length(args)>=6) as.integer(args[6]) else 12
OUTDIR    <- if(length(args)>=7) args[7] else "outputs/dynamicgp_megalmm_innerens"
MAX_INNER <- if(length(args)>=8) as.integer(args[8]) else 5

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

gids <- unique(data$bio_ID)
grm  <- grm[gids, gids]
K    <- as.matrix(Matrix::forceSymmetric(grm))
colnames(K) <- rownames(K) <- gids
lmm_data <- data.frame(Line = gids, row.names = gids)
trait_cols <- colnames(data)[-c(1,2,3)]

# ---- MegaLMM fit/predict: focal Y (train rows observed, val rows NA) ----
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

# ---- one CV iteration (Protocol B): MegaLMM inner-5 ensemble ----
run_iteration <- function(k) {
  cvcol <- cv[, c(1, k+1)]; rownames(cvcol) <- cvcol[, 1]
  folds <- cvcol[gids, 2]
  trn_idx <- which(folds != 6); val_idx <- which(folds == 6)
  trn_gids <- gids[trn_idx]; val_gids <- gids[val_idx]
  cat(sprintf("\n=== iteration %d : train=%d val=%d ===\n", k, length(trn_idx), length(val_idx)))

  # training-fold scaler: min/max fitted on the training subset only,
  # applied to both training and validation (no leakage)
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
  trn_comp <- find_DMD_components(trn_dat, r = R_RANK, method = "schur.DMD", scale = "none")
  trn_R   <- as.matrix(tensor_unroller(trn_comp$R_all))     # train x 4
  trn_phi <- as.matrix(tensor_unroller(trn_comp$phi_all))   # train x 100
  cat("  Schur-DMD: validation subset ...\n")
  val_comp <- find_DMD_components(val_dat, r = R_RANK, method = "schur.DMD", scale = "none")
  val_X   <- val_comp$X_all                                 # 50 x 25 x val
  val_R_true <- as.matrix(tensor_unroller(val_comp$R_all))  # val x 4
  val_phi_true <- as.matrix(tensor_unroller(val_comp$phi_all))  # val x 100
  sv1_cols <- grep("SV1$", colnames(trn_phi)); sv2_cols <- grep("SV2$", colnames(trn_phi))

  kR <- function(nf) min(floor(length(gids)/4), floor(nf/2))
  run_id <- function(m, k, j) sprintf("%s/run_%02d_i%d_%s", OUTDIR, k, j, m)

  # -------- dynamicGP-MegaLMM : inner-5 ensemble (GRM only, no TP1) --------
  inner <- cvcol[trn_gids, 2]      # inner fold label (1..5) of each training gid
  n_inner <- min(5L, MAX_INNER)
  Rl <- list(); P1l <- list(); P2l <- list()
  for (j in 1:5) {
    if (j > n_inner) { cat(sprintf("  [MegaLMM] inner %d: skipped (MAX_INNER=%d)\n", j, n_inner)); next }
    itr <- trn_gids[inner != j]
    cat(sprintf("  [MegaLMM] inner %d: train=%d ...\n", j, length(itr)))
    Rl[[j]]  <- megalmm_predict(trn_R[itr, , drop = FALSE], val_gids,
                                k_factors = kR(4), run_id = run_id("R", k, j))
    P1l[[j]] <- megalmm_predict(trn_phi[itr, sv1_cols, drop = FALSE], val_gids,
                                k_factors = kR(50), run_id = run_id("p1", k, j))
    P2l[[j]] <- megalmm_predict(trn_phi[itr, sv2_cols, drop = FALSE], val_gids,
                                k_factors = kR(50), run_id = run_id("p2", k, j))
  }
  R_mega <- Reduce(`+`, Rl) / length(Rl)
  rownames(R_mega) <- val_gids; colnames(R_mega) <- colnames(trn_R)
  phi_mega <- matrix(NA, length(val_gids), ncol(trn_phi), dimnames = list(val_gids, colnames(trn_phi)))
  phi_mega[, sv1_cols] <- Reduce(`+`, P1l) / length(P1l)
  phi_mega[, sv2_cols] <- Reduce(`+`, P2l) / length(P2l)
  cat(sprintf("  [MegaLMM] inner-5 ensemble done (models used: %d)\n", length(Rl)))

  comp_acc <- function(pred, truth) {
    common <- intersect(rownames(pred), rownames(truth))
    mean(sapply(seq_len(ncol(pred)), function(j) cor(truth[common, j], pred[common, j], use = "complete.obs")), na.rm = TRUE)
  }
  cat(sprintf("  [diag] R comp acc: mega(innerens)=%.3f\n",   comp_acc(R_mega,  val_R_true)))
  cat(sprintf("  [diag] phi comp acc: mega(innerens)=%.3f\n", comp_acc(phi_mega, val_phi_true)))

  evaluate <- function(Rp, PHIp, tag) {
    Rv   <- table_to_tensor(as.data.frame(Rp),   component = "R",   r = R_RANK)
    PHIv <- table_to_tensor(as.data.frame(PHIp), component = "phi", r = R_RANK)
    Av   <- A_r_from_phi_and_R(PHIv, Rv)
    m    <- predict_dynamics(Av, val_X)
    list(iter = as.data.frame(m$iter), rec = as.data.frame(m$rec), pool_iter = m$pool_iter, pool_rec = m$pool_rec)
  }
  cat("  evaluate MegaLMM (inner-5 ensemble) ...\n")
  e_mega <- evaluate(R_mega, phi_mega, "mega")

  list(mega = e_mega)
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