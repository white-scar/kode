# dynamicGP_MegaLMM_CV21_arabidopsis.R
# MegaLMM-CV2.1 on Arabidopsis (45 traits x 14 timepoints). t1 作 secondary，预测 t2-t14。
# 训练折 scaler 归一化。
options(stringsAsFactors = FALSE)
suppressPackageStartupMessages({library(MegaLMM); library(Matrix)})

args <- commandArgs(trailingOnly = TRUE)
START_K <- if(length(args)>=1) as.integer(args[1]) else 1
END_K   <- if(length(args)>=2) as.integer(args[2]) else 1
BURN    <- if(length(args)>=3) as.integer(args[3]) else 300
N_ITER  <- if(length(args)>=4) as.integer(args[4]) else 800
THIN    <- if(length(args)>=5) as.integer(args[5]) else 2
NCORES  <- if(length(args)>=6) as.integer(args[6]) else 8
OUTDIR  <- if(length(args)>=7) args[7] else "outputs/dynamicgp_megalmm_cv21_arab"

setwd("/data/fandata/dashuaige")
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(OMP_NUM_THREADS = as.character(NCORES))
try(suppressWarnings(MegaLMM::set_MegaLMM_nthreads(NCORES)), silent = TRUE)

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
data <- read.csv("data/arabidopsis/BLUE_Experiment_3_IAP_NA_imputed_selected_traits.csv")[, -1]
cv   <- read.csv("data/arabidopsis/cross_validation_folds_w_validation_set_arabidopsis.csv")
grm  <- as.matrix(read.csv("data/arabidopsis/arabidopsis_GRM_382.csv", row.names = 1, check.names = FALSE))

gids <- unique(data$bio_ID)
grm  <- grm[gids, gids]
K    <- as.matrix(Matrix::forceSymmetric(grm)); dimnames(K) <- list(gids, gids)
lmm_data <- data.frame(Line = gids, row.names = gids)
trait_cols <- colnames(data)[-c(1, 2, 3)]
P <- length(trait_cols)
times <- sort(unique(data$DAS)); n_tp <- length(times)

kR <- function(nf) min(floor(length(gids)/4), floor(nf/2))
run_id <- function(m, k) sprintf("%s/run_%02d_%s", OUTDIR, k, m)

for (k in START_K:END_K) {
  cvcol <- cv[, c(1, k+1)]; rownames(cvcol) <- cvcol[, 1]
  folds <- cvcol[gids, 2]
  trn_gids <- gids[folds != 6]; val_gids <- gids[folds == 6]
  cat(sprintf("=== iteration %d train=%d val=%d ===\n", k, length(trn_gids), length(val_gids)))

  trn_data <- data[data$bio_ID %in% trn_gids, ]
  trn_min <- sapply(trn_data[, trait_cols], min, na.rm = TRUE)
  trn_max <- sapply(trn_data[, trait_cols], max, na.rm = TRUE)
  norm_cols <- function(df) {
    df[, trait_cols] <- as.data.frame(mapply(function(x, a, b) (x - a)/(b - a),
                                             df[, trait_cols], trn_min, trn_max, SIMPLIFY = FALSE))
    df
  }
  data_s <- norm_cols(data)

  # focal = t2..t14 的性状；secondary = t1
  focal <- matrix(NA, length(gids), P*(n_tp-1))
  focal_cols <- c()
  for (j in 2:n_tp) focal_cols <- c(focal_cols, paste0(trait_cols, "_t", times[j]))
  colnames(focal) <- focal_cols; rownames(focal) <- gids
  for (j in 2:n_tp) {
    tp <- times[j]
    sub <- data_s[data_s$DAS == tp, ]; rownames(sub) <- sub$bio_ID
    focal[, (j-2)*P + seq_len(P)] <- as.matrix(sub[gids, trait_cols])
  }
  focal[val_gids, ] <- NA

  sub1 <- data_s[data_s$DAS == times[1], ]; rownames(sub1) <- sub1$bio_ID
  secondary <- as.matrix(sub1[gids, trait_cols]); colnames(secondary) <- paste0(trait_cols, "_t1")

  Y <- cbind(focal, secondary)
  nf <- ncol(Y)
  rp <- MegaLMM_control(); rp$K <- kR(nf); rp$h2_divisions <- 20L; rp$burn <- 0L; rp$thin <- THIN
  st <- setup_model_MegaLMM(Y = Y, formula = ~ (1|Line), data = lmm_data,
                            relmat = list(Line = K), run_parameters = rp, run_ID = run_id("cv21", k))
  st <- set_priors_MegaLMM(st, MegaLMM_priors(
    tot_Y_var = list(V = 0.5, nu = 5), tot_F_var = list(V = 18/20, nu = 20),
    h2_priors_resids_fun = function(h2s,n) 1, h2_priors_factors_fun = function(h2s,n) 1,
    Lambda_prior = list(sampler = sample_Lambda_prec_ARD, Lambda_df = 3,
                        delta_1 = list(shape = 2, rate = 1), delta_2 = list(shape = 3, rate = 1),
                        delta_iterations_factor = 100)))
  st <- initialize_variables_MegaLMM(st); st <- fix_megalmm_posterior(st)
  maps <- make_Missing_data_map(st, max_NA_groups = Inf, verbose = FALSE); st <- set_Missing_data_map(st, maps$Missing_data_map)
  st <- initialize_MegaLMM(st, ncores = NCORES)
  for (b in 1:2) { st <- sample_MegaLMM(st, BURN); try(suppressWarnings(st <- reorder_factors(st, drop_cor_threshold = 0.6)), silent=TRUE); st <- clear_Posterior(st); st <- fix_megalmm_posterior(st) }
  st <- sample_MegaLMM(st, N_ITER)
  st <- save_posterior_chunk(st)
  Eta <- load_posterior_param(st, "Eta_mean")
  pred_focal <- Eta[val_gids, seq_len(ncol(focal)), drop = FALSE]

  long <- list()
  for (j in 2:n_tp) {
    tp <- times[j]
    sub <- data_s[data_s$DAS == tp, ]; rownames(sub) <- sub$bio_ID
    truev <- as.matrix(sub[val_gids, trait_cols])
    cols <- (j-2)*P + seq_len(P)
    predv <- pred_focal[, cols, drop = FALSE]
    for (p in seq_len(P)) {
      y <- truev[, p]; pr <- predv[, p]
      ok <- is.finite(y) & is.finite(pr)
      if (sum(ok) < 3) next
      y <- y[ok]; pr <- pr[ok]
      pcc <- cor(y, pr); mse <- mean((y-pr)^2); vy <- var(y)
      nmse <- if(vy>0) mse/vy else NA_real_; r2 <- if(vy>0) 1-mse/vy else NA_real_
      long[[length(long)+1]] <- data.frame(Method="cv21", Iteration=k, Trait=trait_cols[p], Time=paste0("t",j), PCC=pcc, MSE=mse, NMSE=nmse, R2=r2)
    }
  }
  ldf <- do.call(rbind, long)
  write.csv(ldf, file.path(OUTDIR, sprintf("cv21_long_fold_%02d.csv", k)), row.names = FALSE)
  overall <- aggregate(cbind(PCC, MSE, NMSE, R2) ~ Method, data = ldf, FUN = function(x) mean(x, na.rm=TRUE))
  write.csv(overall, file.path(OUTDIR, sprintf("cv21_overall_fold_%02d.csv", k)), row.names = FALSE)
  cat(sprintf("fold %d done\n", k)); print(overall)
}

csvs <- sort(list.files(OUTDIR, pattern = "cv21_long_fold_[0-9]+\\.csv$", full.names = TRUE))
if (length(csvs) > 0) {
  long <- do.call(rbind, lapply(csvs, read.csv))
  write.csv(long, file.path(OUTDIR, "cv21_metrics_long.csv"), row.names = FALSE)
  overall <- aggregate(cbind(PCC, MSE, NMSE, R2) ~ Method, data = long, FUN = function(x) mean(x, na.rm=TRUE))
  write.csv(overall, file.path(OUTDIR, "cv21_metrics_overall.csv"), row.names = FALSE)
  cat("\n=== MegaLMM-CV2.1 overall ===\n"); print(overall)
}
cat("\nALL DONE\n")
