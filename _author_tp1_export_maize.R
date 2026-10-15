# -*- coding: utf-8 -*-
# author_tp1_export_maize.R -- 只跑 tp1+MegaLMM+dynamicGP, 导出每折算子 (train/val true + val predicted)
args <- commandArgs(trailingOnly = TRUE)
START_K <- if(length(args)>=1) as.integer(args[1]) else 1
END_K   <- if(length(args)>=2) as.integer(args[2]) else 10
THIN    <- if(length(args)>=3) as.integer(args[3]) else 2
NCORES  <- if(length(args)>=4) as.integer(args[4]) else 8
OUTDIR  <- if(length(args)>=5) args[5] else "outputs/reproduce/author_tp1_maize"
EXP     <- file.path(OUTDIR, "components")

setwd("/data/fandata/dashuaige")
for (f in c("find_DMD_components.R","tensor_unroller.R","table_to_tensor.R",
            "A_r_from_phi_and_R.R","all_timepoints_predictor.R"))
  source(file.path("reproduce/R", f))
dir.create(EXP, recursive = TRUE, showWarnings = FALSE)
R_RANK <- 2; N_GENO <- 330
Sys.setenv(OMP_NUM_THREADS = as.character(NCORES))
try(suppressWarnings(MegaLMM::set_MegaLMM_nthreads(NCORES)), silent = TRUE)
try(suppressWarnings(MegaLMM::set_omp_nthreads(NCORES)), silent = TRUE)

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
K    <- as.matrix(Matrix::forceSymmetric(grm)); colnames(K) <- rownames(K) <- gids
lmm_data <- data.frame(Line = gids, row.names = gids)
trait_cols <- colnames(data)[-c(1,2,3)]
tp1_raw <- data[data$DAS == 15, ]; rownames(tp1_raw) <- tp1_raw$bio_ID
tp1 <- as.data.frame(lapply(tp1_raw[, trait_cols], function(x) (x - min(x))/(max(x) - min(x))))
rownames(tp1) <- tp1_raw$bio_ID; tp1 <- as.matrix(tp1[gids, , drop = FALSE]); storage.mode(tp1) <- "double"

megalmm_predict <- function(train_mat, val_gids, secondary = NULL, k_factors, run_id) {
  focal <- matrix(NA, length(gids), ncol(train_mat), dimnames = list(gids, colnames(train_mat)))
  focal[rownames(train_mat), ] <- as.matrix(train_mat)
  Y <- focal
  if (!is.null(secondary)) Y <- cbind(focal, secondary[gids, , drop = FALSE])
  Y2 <- Y; Y2[val_gids, seq_len(ncol(focal))] <- NA
  rp <- MegaLMM_control(); rp$K <- k_factors; rp$h2_divisions <- 20L; rp$burn <- 0L; rp$thin <- THIN
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
  st <- run_MegaLMM(st, n_iter = 300, n_burn = 200, verbose = FALSE)
  pred <- st$Posterior$Eta_mean[val_gids, seq_len(ncol(focal)), drop = FALSE]
  rownames(pred) <- val_gids; colnames(pred) <- colnames(train_mat)
  pred
}

for (k in START_K:END_K) {
  cat(sprintf("\n=== iteration %d ===\n", k)); flush.console()
  folds <- cv[, k + 1]
  trn_idx <- which(folds != 6); val_idx <- which(folds == 6)
  trn_gids <- gids[trn_idx]; val_gids <- gids[val_idx]
  trn_dat <- data[data$bio_ID %in% trn_gids, ]; val_dat <- data[data$bio_ID %in% val_gids, ]
  trn_min <- sapply(trn_dat[, -c(1:3)], min, na.rm = TRUE); trn_max <- sapply(trn_dat[, -c(1:3)], max, na.rm = TRUE)
  norm_tr <- function(df, mn, mx) { df[, -c(1:3)] <- as.data.frame(mapply(function(x,a,b)(x-a)/(b-a), df[,-c(1:3)], mn, mx, SIMPLIFY=FALSE)); df }
  trn_dat <- norm_tr(trn_dat, trn_min, trn_max); val_dat <- norm_tr(val_dat, trn_min, trn_max)
  trn_comp <- find_DMD_components(trn_dat, r = R_RANK, method = "schur.DMD", scale = "none")
  trn_R <- as.matrix(tensor_unroller(trn_comp$R_all)); trn_phi <- as.matrix(tensor_unroller(trn_comp$phi_all))
  val_comp <- find_DMD_components(val_dat, r = R_RANK, method = "schur.DMD", scale = "none")
  val_R_true <- as.matrix(tensor_unroller(val_comp$R_all)); val_phi_true <- as.matrix(tensor_unroller(val_comp$phi_all))
  sv1 <- grep("SV1$", colnames(trn_phi)); sv2 <- grep("SV2$", colnames(trn_phi))
  kR <- function(nf) min(floor(length(gids)/4), floor(nf/2))
  rid <- function(m) sprintf("%s/run_%02d_%s", OUTDIR, k, m)
  cat("  tp1 R ...\n"); R_tp1 <- megalmm_predict(trn_R, val_gids, secondary=tp1, k_factors=kR(4+50), run_id=rid("tp1_R"))
  cat("  tp1 phi SV1 ...\n"); p1 <- megalmm_predict(trn_phi[,sv1,drop=FALSE], val_gids, secondary=tp1, k_factors=kR(50+50), run_id=rid("tp1_p1"))
  cat("  tp1 phi SV2 ...\n"); p2 <- megalmm_predict(trn_phi[,sv2,drop=FALSE], val_gids, secondary=tp1, k_factors=kR(50+50), run_id=rid("tp1_p2"))
  phi_tp1 <- matrix(NA, length(val_gids), ncol(trn_phi), dimnames=list(val_gids, colnames(trn_phi)))
  phi_tp1[,sv1] <- p1; phi_tp1[,sv2] <- p2
  write.csv(trn_R,        file.path(EXP, sprintf("iter%02d_trn_R.csv", k)))
  write.csv(trn_phi,      file.path(EXP, sprintf("iter%02d_trn_phi.csv", k)))
  write.csv(val_R_true,   file.path(EXP, sprintf("iter%02d_val_R_true.csv", k)))
  write.csv(val_phi_true, file.path(EXP, sprintf("iter%02d_val_phi_true.csv", k)))
  write.csv(R_tp1,        file.path(EXP, sprintf("iter%02d_val_R_pred.csv", k)))
  write.csv(phi_tp1,      file.path(EXP, sprintf("iter%02d_val_phi_pred.csv", k)))
  cat(sprintf("  saved iter %d components\n", k)); flush.console()
}
cat("\nDONE\n")
