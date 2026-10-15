# dynamicGP_MegaLMM_static_maize_rgb.R
# Pure MegaLMM static genomic prediction for maize UAV RGB (17 traits x 12 flights, Adak 2023).
# Protocol: fold1-5 train / fold6 test (one fold-set per run); GRM K;
#   MegaLMM predicts the P=17 trait vector at each observed timepoint.
# Metrics: PCC and MSE in the TRAIN-fold per-trait min-max space (to match the KNode run).
# Usage: Rscript dynamicGP_MegaLMM_static_maize_rgb.R <env=drought|irrigated> <foldset> <burn> <niter> <thin> <ncores> <outdir>
options(stringsAsFactors = FALSE)
suppressPackageStartupMessages({library(MegaLMM); library(Matrix); library(pracma)})

args <- commandArgs(trailingOnly = TRUE)
ENV    <- if(length(args)>=1) args[1] else "drought"
K_FOLD <- if(length(args)>=2) as.integer(args[2]) else 1
BURN   <- if(length(args)>=3) as.integer(args[3]) else 300
N_ITER <- if(length(args)>=4) as.integer(args[4]) else 800
THIN   <- if(length(args)>=5) as.integer(args[5]) else 2
NCORES <- if(length(args)>=6) as.integer(args[6]) else 8
OUTDIR <- if(length(args)>=7) args[7] else "outputs/kode_result/maize_rgb"

if (dir.exists("/data/fandata/dashuaige")) setwd("/data/fandata/dashuaige") else setwd("D:/pythonproject/knode")
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(OMP_NUM_THREADS = as.character(NCORES))
Sys.setenv(MKL_NUM_THREADS = as.character(NCORES))
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

cat(sprintf("== MegaLMM static | env=%s foldset=%d burn=%d iter=%d thin=%d ncores=%d ==\n", ENV, K_FOLD, BURN, N_ITER, THIN, NCORES))
data <- read.csv(sprintf("data/maize_rgb/maize_rgb_%s_traits.csv", ENV), row.names = 1, check.names = FALSE)
cv   <- read.csv(sprintf("data/maize_rgb/maize_rgb_%s_cv_10fold.csv", ENV), row.names = 1)
grm  <- as.matrix(read.csv("data/maize_rgb/geno_GRM.csv", row.names = 1, check.names = FALSE))

data$bio_ID <- as.character(data$bio_ID)
trait_cols <- setdiff(colnames(data), c("bio_ID", "DAS"))
gids <- unique(data$bio_ID)
grm  <- grm[gids, gids]
K    <- as.matrix(Matrix::forceSymmetric(grm)); dimnames(K) <- list(gids, gids)
lmm_data <- data.frame(Line = gids, row.names = gids)
P <- length(trait_cols)
times <- sort(unique(data$DAS)); n_tp <- length(times)
kfac <- max(2, min(floor(length(gids)/4), floor(P/2)))
cat(sprintf("N=%d P=%d T=%d kfac=%d\n", length(gids), P, n_tp, kfac))

megalmm_static_tp <- function(tp, trn_gids, val_gids, run_id) {
  sub <- data[data$DAS == tp, ]; rownames(sub) <- as.character(sub$bio_ID)
  Yall <- as.matrix(sub[gids, trait_cols])
  trn <- Yall[trn_gids, , drop = FALSE]
  mn <- apply(trn, 2, mean, na.rm = TRUE); sdx <- apply(trn, 2, sd, na.rm = TRUE)
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

out_long <- file.path(OUTDIR, sprintf("megalmm_static_%s_fold%02d.csv", ENV, K_FOLD))
if (file.exists(out_long)) { cat("exists, skip\n"); quit(save="no") }
run_dirs <- list.files(OUTDIR, pattern = sprintf("^run_%s_%02d_", ENV, K_FOLD), full.names = TRUE)
if (length(run_dirs) > 0) unlink(run_dirs, recursive = TRUE)

folds <- cv[gids, K_FOLD]
trn_gids <- gids[folds != 6]; val_gids <- gids[folds == 6]
cat(sprintf("fold-set %d: train=%d val=%d\n", K_FOLD, length(trn_gids), length(val_gids)))

# ---- TRAIN-fold per-trait min-max (same space as the KNode run) ----
trn_data <- data[data$bio_ID %in% trn_gids, ]
trn_min <- apply(trn_data[, trait_cols], 2, min, na.rm = TRUE)
trn_max <- apply(trn_data[, trait_cols], 2, max, na.rm = TRUE)
trn_rng <- pmax(trn_max - trn_min, 1e-8)

long <- list()
t0 <- Sys.time()
for (j in seq_along(times)) {
  tp <- times[j]
  res <- tryCatch({
    pred <- megalmm_static_tp(tp, trn_gids, val_gids, run_id = sprintf("%s/run_%s_%02d_t%02d", OUTDIR, ENV, K_FOLD, j))
    sub <- data[data$DAS == tp, ]; rownames(sub) <- as.character(sub$bio_ID)
    truev <- as.matrix(sub[val_gids, trait_cols])
    for (p in seq_len(P)) {
      y <- truev[, p]; pr <- pred[, p]
      ok <- is.finite(y) & is.finite(pr)
      if (sum(ok) < 3) { long[[length(long)+1]] <- data.frame(Method="megalmm_static", Env=ENV, Iteration=K_FOLD, Trait=trait_cols[p], DAS=tp, PCC=NA_real_, MSE=NA_real_); next }
      y <- y[ok]; pr <- pr[ok]
      pcc <- cor(y, pr)
      yn <- (y - trn_min[p]) / trn_rng[p]; pn <- (pr - trn_min[p]) / trn_rng[p]
      mse <- mean((yn - pn)^2)
      long[[length(long)+1]] <- data.frame(Method="megalmm_static", Env=ENV, Iteration=K_FOLD, Trait=trait_cols[p], DAS=tp, PCC=pcc, MSE=mse)
    }
    sprintf("  t%02d (DAS %d) done  [%.1f min]", j, tp, as.numeric(difftime(Sys.time(), t0, units="mins")))
  }, error = function(e) { cat(sprintf("  t%02d err: %s\n", j, conditionMessage(e))); NULL })
  if (!is.null(res)) { cat(res, "\n"); flush.console() }
}
ldf <- do.call(rbind, long)
write.csv(ldf, out_long, row.names = FALSE)

bt <- aggregate(cbind(PCC, MSE) ~ Env + Method + DAS, data = ldf, FUN = function(x) mean(x, na.rm=TRUE))
tr <- aggregate(cbind(PCC, MSE) ~ Env + Method + Trait, data = ldf, FUN = function(x) mean(x, na.rm=TRUE))
keep <- ldf[ldf$DAS != times[1], ]
ov <- aggregate(cbind(PCC, MSE) ~ Env + Method, data = keep, FUN = function(x) mean(x, na.rm=TRUE))
write.csv(bt, file.path(OUTDIR, sprintf("megalmm_static_%s_by_time.csv", ENV)), row.names = FALSE)
write.csv(tr, file.path(OUTDIR, sprintf("megalmm_static_%s_by_trait.csv", ENV)), row.names = FALSE)
write.csv(ov, file.path(OUTDIR, sprintf("megalmm_static_%s_overall.csv", ENV)), row.names = FALSE)
cat("\n=== overall (excl t1) ===\n"); print(ov)
cat("\nALL DONE\n")
