# dynamicGP_MegaLMM_static_arabidopsis.R
# Pure MegaLMM static genomic prediction (snapshot, per observed timepoint).
# Protocol consistent with dynamicGP_MegaLMM_TP1_arabidopsis.R:
#   folds1-5 train / fold6 test; GRM K; MegaLMM predicts the P=45 trait vector at each timepoint.
#   Metrics per trait x timepoint in the validation min-max [0,1] space (same as dynamicGP snapshot metrics).
# Run one fold first (default fold 4, the RR-BLUP+dGP-representative fold).
options(stringsAsFactors = FALSE)
suppressPackageStartupMessages({library(MegaLMM); library(Matrix); library(pracma)})

args <- commandArgs(trailingOnly = TRUE)
START_K <- if(length(args)>=1) as.integer(args[1]) else 4
END_K   <- if(length(args)>=2) as.integer(args[2]) else 4
BURN    <- if(length(args)>=3) as.integer(args[3]) else 300
N_ITER  <- if(length(args)>=4) as.integer(args[4]) else 800
THIN    <- if(length(args)>=5) as.integer(args[5]) else 2
NCORES  <- if(length(args)>=6) as.integer(args[6]) else 12
OUTDIR  <- if(length(args)>=7) args[7] else "outputs/reproduce/dynamicGP_MegaLMM_static_arabidopsis"

setwd("/data/fandata/dashuaige")
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

Sys.setenv(OMP_NUM_THREADS = as.character(NCORES))
Sys.setenv(MKL_NUM_THREADS = '8')
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
kfac <- min(floor(length(gids)/4), floor(P/2))
cat(sprintf("N=%d P=%d T=%d kfac=%d times=%s\n", length(gids), P, n_tp, kfac, paste(times, collapse=",")))

megalmm_static_tp <- function(tp, trn_gids, val_gids, run_id) {
  sub <- data[data$DAS == tp, ]; rownames(sub) <- sub$bio_ID
  Yall <- as.matrix(sub[gids, trait_cols])
  trn <- Yall[trn_gids, , drop = FALSE]
  mn <- apply(trn, 2, mean); sdx <- apply(trn, 2, sd)
  keep <- which(sdx > 1e-12)
  if (length(keep) == 0) return(matrix(NA, length(val_gids), P, dimnames = list(val_gids, trait_cols)))
  Ztrain <- scale(trn[, keep, drop = FALSE], center = mn[keep], scale = sdx[keep])
  focal <- matrix(NA, length(gids), ncol(Ztrain), dimnames = list(gids, colnames(Ztrain)))
  focal[rownames(Ztrain), ] <- Ztrain
  Y2 <- focal; Y2[val_gids, ] <- NA
  rp <- MegaLMM_control(); rp$K <- kfac; rp$h2_divisions <- 20L; rp$burn <- 0L; rp$thin <- THIN
  st <- setup_model_MegaLMM(Y = Y2, formula = ~ (1|Line), data = lmm_data,
                            relmat = list(Line = K), run_parameters = rp, run_ID = run_id)
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
  pred_keep <- Eta[val_gids, seq_len(ncol(focal)), drop = FALSE]
  pred <- matrix(NA, length(val_gids), P, dimnames = list(val_gids, trait_cols))
  pred[, keep] <- sweep(sweep(pred_keep, 2, sdx[keep], "*"), 2, mn[keep], "+")
  pred
}

for (k in START_K:END_K) {
  out_long <- file.path(OUTDIR, sprintf("static_long_fold_%02d.csv", k))
  if (file.exists(out_long)) { cat(sprintf("skip fold %d (exists)\n", k)); next }
  run_dirs <- list.files(OUTDIR, pattern = sprintf("^run_%02d_", k), full.names = TRUE)
  if (length(run_dirs) > 0) unlink(run_dirs, recursive = TRUE)

  cvcol <- cv[, c(1, k+1)]; rownames(cvcol) <- cvcol[, 1]
  folds <- cvcol[gids, 2]
  trn_gids <- gids[folds != 6]; val_gids <- gids[folds == 6]
  cat(sprintf("=== fold %d train=%d val=%d ===\n", k, length(trn_gids), length(val_gids)))

  val_data <- data[data$bio_ID %in% val_gids, ]
  val_min <- apply(val_data[, trait_cols], 2, min, na.rm = TRUE)
  val_max <- apply(val_data[, trait_cols], 2, max, na.rm = TRUE)
  val_rng <- val_max - val_min; val_rng[val_rng < 1e-12] <- 1

  long <- list()
  for (j in seq_along(times)) {
    tp <- times[j]
    pred <- tryCatch(megalmm_static_tp(tp, trn_gids, val_gids, run_id = sprintf("%s/run_%02d_t%02d", OUTDIR, k, j)),
                     error = function(e) { cat(sprintf("  static err t%s: %s\n", tp, conditionMessage(e))); NULL })
    sub <- data[data$DAS == tp, ]; rownames(sub) <- sub$bio_ID
    truev <- as.matrix(sub[val_gids, trait_cols])
    for (p in seq_len(P)) {
      if (is.null(pred)) {
        long[[length(long)+1]] <- data.frame(Method="static", Iteration=k, Trait=trait_cols[p], Time=paste0("t",j), PCC=NA_real_, MSE=NA_real_, NMSE=NA_real_, R2=NA_real_)
        next
      }
      y <- truev[, p]; pr <- pred[, p]
      ok <- is.finite(y) & is.finite(pr)
      if (sum(ok) < 3) {
        long[[length(long)+1]] <- data.frame(Method="static", Iteration=k, Trait=trait_cols[p], Time=paste0("t",j), PCC=NA_real_, MSE=NA_real_, NMSE=NA_real_, R2=NA_real_)
        next
      }
      y <- y[ok]; pr <- pr[ok]
      pcc <- cor(y, pr)
      yn <- (y - val_min[p]) / val_rng[p]; pn <- (pr - val_min[p]) / val_rng[p]
      mse <- mean((yn - pn)^2); vy <- var(yn)
      nmse <- if (vy > 0) mse/vy else NA_real_
      r2 <- if (vy > 0) 1 - mse/vy else NA_real_
      long[[length(long)+1]] <- data.frame(Method="static", Iteration=k, Trait=trait_cols[p], Time=paste0("t",j), PCC=pcc, MSE=mse, NMSE=nmse, R2=r2)
    }
  }
  ldf <- do.call(rbind, long)
  write.csv(ldf, out_long, row.names = FALSE)
  cat(sprintf("fold %d done\n", k))
}

csv_files <- sort(list.files(OUTDIR, pattern = "static_long_fold_[0-9]+\\.csv$", full.names = TRUE))
if (length(csv_files) > 0) {
  long <- do.call(rbind, lapply(csv_files, read.csv))
  write.csv(long, file.path(OUTDIR, "snapshot_metrics_long.csv"), row.names = FALSE)
  by_time <- aggregate(cbind(PCC, MSE, NMSE, R2) ~ Method + Time, data = long, FUN = function(x) mean(x, na.rm=TRUE))
  write.csv(by_time, file.path(OUTDIR, "snapshot_metrics_by_time.csv"), row.names = FALSE)
  keep <- long[long$Time != "t1", ]
  overall <- aggregate(cbind(PCC, MSE, NMSE, R2) ~ Method, data = keep, FUN = function(x) mean(x, na.rm=TRUE))
  write.csv(overall, file.path(OUTDIR, "snapshot_metrics_overall.csv"), row.names = FALSE)
  pool <- aggregate(cbind(PCC, MSE, NMSE, R2) ~ Method + Iteration, data = keep, FUN = function(x) mean(x, na.rm=TRUE))
  write.csv(pool, file.path(OUTDIR, "snapshot_metrics_pooled.csv"), row.names = FALSE)
  cat("\n=== static MegaLMM overall (mean over traits x timepoints, exclude t1) ===\n")
  print(overall)
  cat("\nALL DONE\n")
} else {
  cat("no fold CSVs found\n")
}
