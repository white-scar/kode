
# ============================================================================
# dynamicGP-MegaLMM and dynamicGP-MegaLMM+TP1 (maize) reproduction
# Protocol (unified, no leakage): scaler fitted on the TRAINING fold only
# (per-trait min/max over folds1-5), applied to BOTH train and validation;
# DMD is called with scale="none" so there is no second normalization.
# NOTE: the author's original code used find_DMD_components() default
# scale="min.max" on training_data / validation_data SEPARATELY (per-subset),
# which leaks the validation range; that is deliberately NOT used here.
#
# Metrics (snapshot, per trait x timepoint across validation genotypes):
#   PCC, MSE, NMSE (= MSE / var(y_true)), R2 (= 1 - NMSE)
#
# Methods compared (same DMD components, different core GP):
#   1. dynamicGP-MegaLMM      : MegaLMM (GRM) predicts R and the two phi vectors
#   2. dynamicGP-MegaLMM+TP1  : same + all 50 t=1 traits as secondary traits
#   3. dynamicGP-RR-BLUP      : rrBLUP (SNP markers) predicts R and phi (baseline)
# ============================================================================
#
# ============================================================================
# [PRESERVED FROM dynamicGP_maize.R] Heritability analysis (kept for reference
#  only; NOT executed by this script). Includes the GCTA placeholder (returns
#  NA), the per-component SVD/Schur heritability tables, and Figure S14/S15.
# ============================================================================
#
# GCTA <- function(matrix_unrolled, kinship_filename = "", wd = "") {
#   h <- rep(NA, ncol(matrix_unrolled))
#   names(h) <- colnames(matrix_unrolled)
#   return(h)
# }
#
# ########################################################################################################################
# # Heritability Analysis
# ########################################################################################################################
#
# data <- read.csv("data/maize_magic/maize_trait_BLUP_50_NArmv_traits_afterClustering.csv")[, -1]
# days <- unique(data$DAS)
#
# cv <- read.csv("data/maize_magic/cross_validation_folds_w_validation_set_maize.csv")
#
# iterations <- 10
#
# validation_heritability <- training_heritability <- data.frame()
#
# for (k in 1:10) {
#   cv_loop <- cv[, c(1, k + 1)]
#   rownames(cv_loop) <- cv_loop[, 1]
#
#   validation_data <- data[data$bio_ID %in% cv_loop$X[which(cv_loop$cv_idx == 6)], ]
#   validation_set_components <- find_DMD_components(validation_data, r = 2, method = "schur.DMD")
#
#   validation_U <- tensor_unroller(validation_set_components$U_r_transpose_all)
#   # write.csv(validation_U, "outputs/dynamicGP/Unrolled_matrices/Validation_set/U_r_unrolled_r=2_validation_set.csv")
#   validation_U_h <- cbind.data.frame("U", "Validation", 2, k, GCTA(validation_U))
#   validation_U_h$Element <- rownames(validation_U_h)
#   rownames(validation_U_h) <- NULL
#   colnames(validation_U_h) <- c("Component", "TV", "r", "Iteration", "Heritablity", "Element")
#   # write.csv(validation_U_h, "outputs/dynamicGP/Heritability/Validation_set/U_r_unrolled_r=2_validation_set_h.csv")
#
#   validation_V <- tensor_unroller(validation_set_components$V_r_all)
#   # write.csv(validation_V, "outputs/dynamicGP/Unrolled_matrices/Validation_set/V_unrolled_r=2_validation_set.csv")
#   validation_V_h <- cbind.data.frame("V", "Validation", 2, k, GCTA(validation_V))
#   validation_V_h$Element <- rownames(validation_V_h)
#   rownames(validation_V_h) <- NULL
#   colnames(validation_V_h) <- c("Component", "TV", "r", "Iteration", "Heritablity", "Element")
#   # write.csv(validation_V_h, "outputs/dynamicGP/Heritability/Validation_set/V_unrolled_r=2_validation_set_h.csv")
#
#   validation_sigma <- tensor_unroller(validation_set_components$sigma_r_all)
#   # write.csv(validation_sigma, "outputs/dynamicGP/Unrolled_matrices/Validation_set/sigma_unrolled_r=2_validation_set.csv")
#   validation_sigma_h <- cbind.data.frame("sigma", "Validation", 2, k, GCTA(validation_sigma))
#   validation_sigma_h$Element <- rownames(validation_sigma_h)
#   rownames(validation_sigma_h) <- NULL
#   colnames(validation_sigma_h) <- c("Component", "TV", "r", "Iteration", "Heritablity", "Element")
#   # write.csv(validation_sigma_h, "outputs/dynamicGP/Heritability/Validation_set/sigma_unrolled_r=2_validation_set_h.csv")
#
#   validation_A_tilde <- tensor_unroller(validation_set_components$A_tilde_all)
#   # write.csv(validation_A_tilde, "outputs/dynamicGP/Unrolled_matrices/Validation_set/A_tilde_unrolled_r=2_validation_set.csv")
#   validation_A_tilde_h <- cbind.data.frame("A_tilde", "Validation", 2, k, GCTA(validation_A_tilde))
#   validation_A_tilde_h$Element <- rownames(validation_A_tilde_h)
#   rownames(validation_A_tilde_h) <- NULL
#   colnames(validation_A_tilde_h) <- c("Component", "TV", "r", "Iteration", "Heritablity", "Element")
#   # write.csv(validation_A_tilde_h, "outputs/dynamicGP/Heritability/Validation_set/A_tilde_unrolled_r=2_validation_set_h.csv")
#
#   validation_Q <- tensor_unroller(validation_set_components$Q_all)
#   # write.csv(validation_Q, "outputs/dynamicGP/Unrolled_matrices/Validation_set/Q_unrolled_r=2_validation_set.csv")
#   validation_Q_h <- cbind.data.frame("Q", "Validation", 2, k, GCTA(validation_Q))
#   validation_Q_h$Element <- rownames(validation_Q_h)
#   rownames(validation_Q_h) <- NULL
#   colnames(validation_Q_h) <- c("Component", "TV", "r", "Iteration", "Heritablity", "Element")
#   # write.csv(validation_Q_h, "outputs/dynamicGP/Heritability/Validation_set/Q_unrolled_r=2_validation_set_h.csv")
#
#   validation_R <- tensor_unroller(validation_set_components$R_all)
#   # write.csv(validation_R, "outputs/dynamicGP/Unrolled_matrices/Validation_set/R_unrolled_r=2_validation_set.csv")
#   validation_R_h <- cbind.data.frame("R", "Validation", 2, k, GCTA(validation_R))
#   validation_R_h$Element <- rownames(validation_R_h)
#   rownames(validation_R_h) <- NULL
#   colnames(validation_R_h) <- c("Component", "TV", "r", "Iteration", "Heritablity", "Element")
#   # write.csv(validation_R_h, "outputs/dynamicGP/Heritability/Validation_set/R_unrolled_r=2_validation_set_h.csv")
#
#   validation_phi <- tensor_unroller(validation_set_components$phi_all)
#   # write.csv(validation_phi, "outputs/dynamicGP/Unrolled_matrices/Validation_set/phi_unrolled_r=2_validation_set.csv")
#   validation_phi_h <- cbind.data.frame("phi", "Validation", 2, k, GCTA(validation_phi))
#   validation_phi_h$Element <- rownames(validation_phi_h)
#   rownames(validation_phi_h) <- NULL
#   colnames(validation_phi_h) <- c("Component", "TV", "r", "Iteration", "Heritablity", "Element")
#   # write.csv(validation_phi_h, "outputs/dynamicGP/Heritability/Validation_set/phi_unrolled_r=2_validation_set_h.csv")
#
#   validation_heritability <- rbind.data.frame(validation_heritability, validation_U_h, validation_V_h, validation_sigma_h, validation_A_tilde_h, validation_Q_h, validation_R_h, validation_phi_h)
#
#   training_data <- data[data$bio_ID %in% cv_loop$X[which(cv_loop$cv_idx != 6)], ]
#
#   for (i in 2:6) {
#     training_components <- find_DMD_components(training_data, r = i, method = "schur.DMD")
#
#     # dir.create(paste0("outputs/dynamicGP/Unrolled_matrices/Training_set/r=", i))
#     # dir.create(paste0("outputs/dynamicGP/Heritability/Training_set/r=", i))
#
#     if (i == 6) {
#       training_U <- tensor_unroller(training_components$U_r_transpose_all)
#       # write.csv(training_U, paste0("outputs/dynamicGP/Unrolled_matrices/Training_set/r=", i, "/U_r_unrolled_r=", i, "_Training_set.csv"))
#       training_U_h <- cbind.data.frame("U", "Training", i, k, GCTA(training_U))
#       # write.csv(training_U_h, paste0("outputs/dynamicGP/Heritability/Training_set/r=", i, "/U_r_unrolled_r=", i, "_Training_set_h.csv"))
#       training_U_h$Element <- rownames(training_U_h)
#       rownames(training_U_h) <- NULL
#       colnames(training_U_h) <- c("Component", "TV", "r", "Iteration", "Heritablity", "Element")
#
#       training_V <- tensor_unroller(training_components$V_r_all)
#       # write.csv(training_V, paste0("outputs/dynamicGP/Unrolled_matrices/Training_set/r=", i, "/V_unrolled_r=", i, "_Training_set.csv"))
#       training_V_h <- cbind.data.frame("V", "Training", i, k, GCTA(training_V))
#       # write.csv(training_V_h, paste0("outputs/dynamicGP/Heritability/Training_set/r=", i, "/V_unrolled_r=", i, "_Training_set_h.csv"))
#       training_V_h$Element <- rownames(training_V_h)
#       rownames(training_V_h) <- NULL
#       colnames(training_V_h) <- c("Component", "TV", "r", "Iteration", "Heritablity", "Element")
#
#       training_sigma <- tensor_unroller(training_components$sigma_r_all)
#       # write.csv(training_sigma, paste0("outputs/dynamicGP/Unrolled_matrices/Training_set/r=", i, "/sigma_unrolled_r=", i, "_Training_set.csv"))
#       training_sigma_h <- cbind.data.frame("sigma", "Training", i, k, GCTA(training_sigma))
#       # write.csv(training_sigma_h, paste0("outputs/dynamicGP/Heritability/Training_set/r=", i, "/sigma_unrolled_r=", i, "_Training_set_h.csv"))
#       training_sigma_h$Element <- rownames(training_sigma_h)
#       rownames(training_sigma_h) <- NULL
#       colnames(training_sigma_h) <- c("Component", "TV", "r", "Iteration", "Heritablity", "Element")
#
#       training_A_tilde <- tensor_unroller(training_components$A_tilde_all)
#       # write.csv(training_A_tilde, paste0("outputs/dynamicGP/Unrolled_matrices/Training_set/r=", i, "/A_tilde_unrolled_r=", i, "_Training_set.csv"))
#       training_A_tilde_h <- cbind.data.frame("A_tilde", "Training", i, k, GCTA(training_A_tilde))
#       # write.csv(training_A_tilde_h, paste0("outputs/dynamicGP/Heritability/Training_set/r=", i, "/A_tilde_unrolled_r=", i, "_Training_set_h.csv"))
#       training_A_tilde_h$Element <- rownames(training_A_tilde_h)
#       rownames(training_A_tilde_h) <- NULL
#       colnames(training_A_tilde_h) <- c("Component", "TV", "r", "Iteration", "Heritablity", "Element")
#   }
#
#     training_Q <- tensor_unroller(training_components$Q_all)
#     # write.csv(training_Q, paste0("outputs/dynamicGP/Unrolled_matrices/Training_set/r=", i, "/Q_unrolled_r=", i, "_Training_set.csv"))
#     training_Q_h <- cbind.data.frame("Q", "Training", i, k, GCTA(training_Q))
#     # write.csv(training_Q_h, paste0("outputs/dynamicGP/Heritability/Training_set/r=", i, "/Q_unrolled_r=", i, "_Training_set_h.csv"))
#     training_Q_h$Element <- rownames(training_Q_h)
#     rownames(training_Q_h) <- NULL
#     colnames(training_Q_h) <- c("Component", "TV", "r", "Iteration", "Heritablity", "Element")
#
#     training_R <- tensor_unroller(training_components$R_all)
#     # write.csv(training_R, paste0("outputs/dynamicGP/Unrolled_matrices/Training_set/r=", i, "/R_unrolled_r=", i, "_Training_set.csv"))
#     training_R_h <- cbind.data.frame("R", "Training", i, k, GCTA(training_R))
#     # write.csv(training_R_h, paste0("outputs/dynamicGP/Heritability/Training_set/r=", i, "/R_unrolled_r=", i, "_Training_set_h.csv"))
#     training_R_h$Element <- rownames(training_R_h)
#     rownames(training_R_h) <- NULL
#     colnames(training_R_h) <- c("Component", "TV", "r", "Iteration", "Heritablity", "Element")
#
#     training_phi <- tensor_unroller(training_components$phi_all)
#     # write.csv(training_phi, paste0("outputs/dynamicGP/Unrolled_matrices/Training_set/r=", i, "/phi_unrolled_r=", i, "_Training_set.csv"))
#     training_phi_h <- cbind.data.frame("phi", "Training", i, k, GCTA(training_phi))
#     # write.csv(training_phi_h, paste0("outputs/dynamicGP/Heritability/Training_set/r=", i, "/phi_unrolled_r=", i, "_Training_set_h.csv"))
#     training_phi_h$Element <- rownames(training_phi_h)
#     rownames(training_phi_h) <- NULL
#     colnames(training_phi_h) <- c("Component", "TV", "r", "Iteration", "Heritablity", "Element")
#
#     if (i == 6) {
#       training_heritability <- rbind.data.frame(training_heritability, training_U_h, training_V_h, training_sigma_h, training_A_tilde_h, training_Q_h, training_R_h, training_phi_h)
#     } else {
#       training_heritability <- rbind.data.frame(training_heritability, training_Q_h, training_R_h, training_phi_h)
#     }
#   }
# }
#
# heritability <- rbind.data.frame(validation_heritability, training_heritability)
# # write.csv(heritability, "outputs/dynamicGP/Heritability/Heritability_all_folds_10iterations_2.0.csv")
#
#
# heritability_file <- "outputs/dynamicGP/Heritability/Heritability_all_folds_10iterations_2.0.csv"
# if (file.exists(heritability_file)) { heritability <- read.csv(heritability_file)
# heritability <- heritability[which(heritability$TV != "Validation"), -1]
#
# heritability$Method <- "Schur"
# heritability$Method[which(heritability$Component == "U")] <- "SVD"
# heritability$Method[which(heritability$Component == "V")] <- "SVD"
# heritability$Method[which(heritability$Component == "sigma")] <- "SVD"
# heritability$Method[which(heritability$Component == "A_tilde")] <- "SVD"
#
# last <- substr(heritability$Element, stringr::str_length(heritability$Element) - 2, stringr::str_length(heritability$Element) - 1)
# first <- substr(heritability$Element, 1, 2)
#
# heritability$R <- NA
# for (i in 1:nrow(heritability)) {
#   if (heritability$Method[i] == "SVD") {
#     if (first[i] == "SV" && last[i] == "SV") {
#       heritability$R[i] <- as.numeric(max(substr(heritability$Element[i], 3, 3), substr(heritability$Element[i], stringr::str_length(heritability$Element[i]), stringr::str_length(heritability$Element[i]))))
#     } else if (first[i] == "SV" && last[i] != "SV") {
#       heritability$R[i] <- as.numeric(substr(heritability$Element[i], 3, 3))
#     } else if (first[i] != "SV" && last[i] == "SV") {
#       heritability$R[i] <- as.numeric(substr(heritability$Element[i], stringr::str_length(heritability$Element[i]), stringr::str_length(heritability$Element[i])))
#     }
#   } else {
#     heritability$R[i] <- heritability$r[i]
#   }
# }
#
# heritability$r <- heritability$R
# heritability <- heritability[, -8]
# heritability <- heritability[-which(is.na(heritability$Heritability) == TRUE), ]
#
#
# title_size <- 7
# axis_text_size <- 7
#
# heritability$r <- factor(heritability$r, levels = sort(unique(heritability$r)))
# heritability$Component[which(heritability$Component == "A_tilde")] <- "\U00C3"
# heritability$Component[which(heritability$Component == "sigma")] <- "\U03A3"
# heritability$Component[which(heritability$Component == "phi")] <- "\U03D5"
#
# library("ggplot2")
#
# figS14 <- ggplot(data = heritability[which(heritability$Method == "SVD"), ], aes(y = Heritability, x = r, fill = Component)) +
#   # geom_violin()
#   geom_boxplot(position = position_dodge(), lwd = 0.25, outlier.size = 0.05) +
#   scale_fill_manual(values = c('#00120B', '#6B818C', '#709BFF', "#17DE6D")) + # '#00120B',
#   xlab("Corresponding singular vector") +
#   theme_classic() +
#   theme(legend.position = c(0.85, 0.85),
#         # legend.box.background = element_rect(colour = "black"),
#         legend.key.width = unit(0.35, 'cm'),
#         legend.key.height = unit(0.35, "cm"),
#         legend.title = element_text(size = 5),
#         legend.text = element_text(size = 5),
#         axis.title = element_text(size = title_size, face = "plain"),
#         axis.text = element_text(size = axis_text_size)
#   )
#
# figS14
#
# ggsave(filename = "outputs/reproduce/Figure S14 - Heritability of SVD components - maize - after validation.eps",
#        device = "eps",
#        plot = figS14,
#        width = 88,
#        height = 88,
#        units = "mm",
#        dpi = 1000
# )
#
#
# heritability$Component <- factor(heritability$Component, levels = c("\U03D5", "Q", "R", "\U00C3", "U", "V", "\U03A3"))
#
# figS15 <- ggplot(data = heritability[which(heritability$Method == "Schur"), ], aes(y = Heritability, x = r, fill = Component)) +
#   # geom_violin()
#   geom_boxplot(position = position_dodge(), lwd = 0.25, outlier.size = 0.05) +
#   scale_fill_manual(values = c('#00120B', '#6B818C', '#709BFF', "#17DE6D")) + # '#00120B',
#   xlab("r") +
#   theme_classic() +
#   theme(legend.position = c(0.85, 0.85),
#         # legend.box.background = element_rect(colour = "black"),
#         legend.key.width = unit(0.35, 'cm'),
#         legend.key.height = unit(0.35, "cm"),
#         legend.title = element_text(size = 5),
#         legend.text = element_text(size = 5),
#         axis.title = element_text(size = title_size, face = "plain"),
#         axis.text = element_text(size = axis_text_size)
#   )
#
# figS15
#
# ggsave(filename = "outputs/reproduce/Figure S15 - Heritability of Schur components - maize - after validation.eps",
#        device = "eps",
#        plot = figS15,
#        width = 88,
#        height = 88,
#        units = "mm",
#        dpi = 1000
# )
#
#
# }
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
OUTDIR  <- if(length(args)>=7) args[7] else "outputs/dynamicgp_megalmm"

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
run_iteration <- function(k) {
  cvcol <- cv[, c(1, k+1)]; rownames(cvcol) <- cvcol[, 1]
  folds <- cvcol[gids, 2]
  trn_idx <- which(folds != 6); val_idx <- which(folds == 6)
  trn_gids <- gids[trn_idx]; val_gids <- gids[val_idx]
  cat(sprintf("\n=== iteration %d : train=%d val=%d ===\n", k, length(trn_idx), length(val_idx)))

  # 训练折 scaler：min/max 只在训练子集上拟合，应用到训练与验证（无泄露）
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
# ---- maize rrBLUP benchmark aggregation (separate benchmark CSV) ----
res_benchmark <- data.frame()
for (k in 1:10) {
  bfile <- paste0("outputs/dynamicGP/Training_Validation_Results/rrBLUP_benchmark_Validation_Prediction_Accuracy/Iteration_", k,
                  "/rrBLUP_benchmark_Aggregated_Prediction_Accuracy_iteration_", k, ".csv")
  if (file.exists(bfile)) {
    res <- read.csv(bfile)
    res_benchmark <- rbind.data.frame(res_benchmark, res)
  }
}
if (nrow(res_benchmark) > 0) {
  res_benchmark <- cbind.data.frame(res_benchmark[, 2:3],
                                    stringr::str_split_fixed(res_benchmark[, 4], "_-_-_", 2),
                                    res_benchmark[, 5:6])
  res_benchmark[, 4] <- substr(res_benchmark[, 4], 2, 3)
  colnames(res_benchmark) <- c("Method", "Iteration", "Trait", "Time", "Accuracy", "MSE")
  write.csv(res_benchmark, "outputs/dynamicGP/Training_Validation_Results/rrBLUP_benchmarks_prediction_accuracies.csv")
}

cat("\nDONE\n")
