# knode_rgb_megalmm_static.R
# MegaLMM static (per observed timepoint) genomic prediction for maize UAV RGB (Adak et al. 2023).
# Adapted from dynamicGP_MegaLMM_static_maize.R to the RGB dataset:
#   - 17 traits (cross-env Mantel>=0.97 clustering + max-h2), 12 flights (DAP 42..99)
#   - same 10 fold-sets as knode_maize_rgb_drought_np_tp1_affine.py (fold1-5 train / fold6 test)
#   - metrics: PCC + MSE, MSE computed in the TRAIN-fold per-trait min-max space (same as KNode)
# Usage: Rscript knode_rgb_megalmm_static.R <env> <fold> [burn] [niter] [thin] [ncores] [outdir]
options(stringsAsFactors = FALSE)
suppressPackageStartupMessages({library(MegaLMM); library(Matrix)})

args <- commandArgs(trailingOnly = TRUE)
ENV    <- if(length(args)>=1) args[1] else "drought"
FOLD   <- if(length(args)>=2) as.integer(args[2]) else 1
BURN   <- if(length(args)>=3) as.integer(args[3]) else 300
N_ITER <- if(length(args)>=4) as.integer(args[4]) else 800
THIN   <- if(length(args)>=5) as.integer(args[5]) else 2
NCORES <- if(length(args)>=6) as.integer(args[6]) else 8
OUTDIR <- if(length(args)>=7) args[7] else sprintf("outputs/kode_result/maize_rgb/megalmm_static_%s", ENV)

TRAITS  <- c("BCC","BRVI","COM1","EBI","ExG","ExGR","G-B","G-R","GCC","MExG","MRCC","NRMBI","R-B","R/B","RCC","RGBVI","TGI")
FLIGHTS <- c(42,56,61,64,68,71,75,78,82,84,90,99)

dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(OMP_NUM_THREADS = as.character(NCORES))
Sys.setenv(MKL_NUM_THREADS = '4')
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

cat(sprintf("env=%s fold=%d burn=%d niter=%d thin=%d ncores=%d\n", ENV, FOLD, BURN, N_ITER, THIN, NCORES))
data <- read.csv(sprintf("data/maize_rgb/maize_rgb_%s_traits.csv", ENV), row.names = 1)
cv   <- read.csv(sprintf("data/maize_rgb/maize_rgb_%s_cv_10fold.csv", ENV), row.names = 1)
grm  <- as.matrix(read.csv("data/maize_rgb/geno_GRM.csv", row.names = 1, check.names = FALSE))
data$bio_ID <- as.character(data$bio_ID)
gids <- sort(unique(data$bio_ID))
P    <- length(TRAITS); n_tp <- length(FLIGHTS)
grm  <- grm[gids, gids]
K    <- as.matrix(Matrix::forceSymmetric(grm)); dimnames(K) <- list(gids, gids)
lmm_data <- data.frame(Line = gids, row.names = gids)
kfac <- min(floor(length(gids)/4), P)
cat(sprintf("N=%d P=%d T=%d kfac=%d\n", length(gids), P, n_tp, kfac))

megalmm_static_tp <- function(tp, trn_gids, val_gids, run_id) {
  sub <- data[data$DAS == tp, ]; rownames(sub) <- sub$bio_ID
  Yall <- as.matrix(sub[gids, TRAITS])
  trn  <- Yall[trn_gids, , drop = FALSE]
  mn <- apply(trn, 2, mean); sdx <- apply(trn, 2, sd)
  keep <- which(sdx > 1e-12)
  if (length(keep) == 0) return(matrix(NA, length(val_gids), P, dimnames = list(val_gids, TRAITS)))
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
  pred <- matrix(NA, length(val_gids), P, dimnames = list(val_gids, TRAITS))
  pred[, keep] <- sweep(sweep(pred_keep, 2, sdx[keep], "*"), 2, mn[keep], "+")
  pred
}

fc <- cv[gids, FOLD]
trn_gids <- gids[fc != 6]; val_gids <- gids[fc == 6]
cat(sprintf("=== fold %d train=%d val=%d ===\n", FOLD, length(trn_gids), length(val_gids)))

long <- list()
for (j in seq_along(FLIGHTS)) {
  tp <- FLIGHTS[j]
  t0 <- Sys.time()
  pred <- tryCatch(megalmm_static_tp(tp, trn_gids, val_gids, run_id = sprintf("%s/run_f%02d_t%02d", OUTDIR, FOLD, j)),
                   error = function(e) { cat(sprintf("  err t%s: %s\n", tp, conditionMessage(e))); NULL })
  sub <- data[data$DAS == tp, ]; rownames(sub) <- sub$bio_ID
  # TRAIN-fold per-trait min-max (same space as KNode)
  tr_min <- apply(as.matrix(sub[trn_gids, TRAITS]), 2, min, na.rm = TRUE)
  tr_max <- apply(as.matrix(sub[trn_gids, TRAITS]), 2, max, na.rm = TRUE)
  tr_rng <- pmax(tr_max - tr_min, 1e-8)
  truev <- as.matrix(sub[val_gids, TRAITS])
  for (p in seq_len(P)) {
    if (is.null(pred)) {
      long[[length(long)+1]] <- data.frame(env=ENV, method="megalmm_static", fold=FOLD, DAS=tp, trait=TRAITS[p], PCC=NA_real_, MSE=NA_real_); next
    }
    y <- truev[, p]; pr <- pred[, p]
    ok <- is.finite(y) & is.finite(pr)
    if (sum(ok) < 3) { long[[length(long)+1]] <- data.frame(env=ENV, method="megalmm_static", fold=FOLD, DAS=tp, trait=TRAITS[p], PCC=NA_real_, MSE=NA_real_); next }
    y <- y[ok]; pr <- pr[ok]
    pcc <- cor(y, pr)
    yn <- (y - tr_min[p]) / tr_rng[p]; pn <- (pr - tr_min[p]) / tr_rng[p]
    mse <- mean((yn - pn)^2)
    long[[length(long)+1]] <- data.frame(env=ENV, method="megalmm_static", fold=FOLD, DAS=tp, trait=TRAITS[p], PCC=pcc, MSE=mse)
  }
  els <- as.numeric(difftime(Sys.time(), t0, units="secs"))
  cat(sprintf("  DAP %d done (%.0fs)\n", tp, els))
}
ldf <- do.call(rbind, long)
write.csv(ldf, file.path(OUTDIR, sprintf("megalmm_static_fold%02d.csv", FOLD)), row.names = FALSE)
cat("\n=== summary (excl t1 = DAP42) ===\n")
k <- ldf[ldf$DAS != FLIGHTS[1], ]
cat(sprintf("PCC=%.4f  MSE=%.6f  (mean over trait x timepoint x fold)\n", mean(k$PCC, na.rm=TRUE), mean(k$MSE, na.rm=TRUE)))
cat("ALL DONE\n")
