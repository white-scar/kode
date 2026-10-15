# -*- coding: utf-8 -*-
# dynamicGP-MegaLMM(+TP1) ? ?? indica control, in-window, inner-5 ??
# ??(???/ Kode ??):
#   * ?: rice_cv_10fold.csv, ?? 1-5 ?? / 6 ??; ?? 10 fold-set
#   * ??: inner-5 (???? 4/5 ?? + 1/5 ????), 5 ???? val ?????????
#   * ???: per-trait min-max ?? **?????? x ????** ??, ???????? (???)
#   * DMD: ????? (find_DMD_components2: ?? DAS ? != 1 ?????; ????? 28->30)
# ??: <OUTDIR>/components/iter%02d_val_{R,phi}_enspred_%s.csv  (variant: mega / tp1)
#       <OUTDIR>/components/iter%02d_val_X_norm.csv               (????????, ? Python ??/??)
# ??: Rscript dynamicGP_MegaLMM_TP1_rice.R START_K END_K THIN NCORES OUTDIR [DATASET]
args <- commandArgs(trailingOnly=TRUE)
START_K <- if(length(args)>=1) as.integer(args[1]) else 1
END_K   <- if(length(args)>=2) as.integer(args[2]) else 10
THIN    <- if(length(args)>=3) as.integer(args[3]) else 2
NCORES  <- if(length(args)>=4) as.integer(args[4]) else 8
OUTDIR  <- if(length(args)>=5) args[5] else "outputs/reproduce/author_dgp_rice_control"
DATASET <- if(length(args)>=6) args[6] else "control"
dir.create(file.path(OUTDIR,"components"), recursive=TRUE, showWarnings=FALSE)
COMP <- file.path(OUTDIR,"components")
setwd("/data/fandata/dashuaige")
for (f in c("find_DMD_components2.R","tensor_unroller.R","table_to_tensor.R","A_r_from_phi_and_R.R"))
  source(file.path("reproduce/R", f))
suppressPackageStartupMessages(library(MegaLMM))
R_RANK <- 2; BURN <- 200; N_ITER <- 300
Sys.setenv(OMP_NUM_THREADS=as.character(NCORES))
try(suppressWarnings(MegaLMM::set_MegaLMM_nthreads(NCORES)), silent=TRUE)
try(suppressWarnings(MegaLMM::set_omp_nthreads(NCORES)), silent=TRUE)
fix_megalmm_posterior <- function(st){ for(p in st$Posterior$posteriorSample_params){ if(is.null(st$Posterior[[p]])&&!is.null(st$current_state[[p]])&&length(dim(st$current_state[[p]]))==2){ st$Posterior[[p]]<-array(0,dim=c(0,dim(st$current_state[[p]]))); dimnames(st$Posterior[[p]])[2:3]<-dimnames(st$current_state[[p]])}}; for(p in st$Posterior$posteriorMean_params){ if(is.null(st$Posterior[[p]])&&!is.null(st$current_state[[p]])&&length(dim(st$current_state[[p]]))==2){ st$Posterior[[p]]<-array(0,dim=dim(st$current_state[[p]])); dimnames(st$Posterior[[p]])<-dimnames(st$current_state[[p]])}}; st }

pheno_f <- if(DATASET=="salt") "rice_indica_salt_traits.csv" else "rice_indica_control_traits.csv"
data <- read.csv(file.path("data/rice_salinity", pheno_f), row.names=1)
cv   <- read.csv("data/rice_salinity/rice_cv_10fold.csv", row.names=1)
grm  <- as.matrix(read.csv("data/rice_salinity/rice_GRM_226x226.csv", row.names=1, check.names=FALSE))
data$bio_ID <- as.character(data$bio_ID); rownames(cv) <- as.character(rownames(cv)); rownames(grm) <- colnames(grm) <- as.character(rownames(grm))
gids <- intersect(intersect(unique(data$bio_ID), rownames(cv)), rownames(grm))
cat(sprintf("dataset=%s  n_gids=%d  n_traits=%d  n_DAS=%d\n", DATASET, length(gids), ncol(data)-3, length(unique(data$DAS))))
grm <- grm[gids,gids]
K <- as.matrix(Matrix::forceSymmetric(grm)); colnames(K)<-rownames(K)<-gids
lmm_data <- data.frame(Line=gids, row.names=gids); trait_cols <- colnames(data)[-c(1,2,3)]
DAS_ALL <- sort(unique(data$DAS))
# TP1 = ?????????? (??????? -> ???), ??? min-max ???
tp1_raw <- data[data$DAS==min(data$DAS),]; rownames(tp1_raw) <- tp1_raw$bio_ID
tp1 <- as.data.frame(lapply(tp1_raw[,trait_cols], function(x)(x-min(x))/(max(x)-min(x))))
rownames(tp1) <- tp1_raw$bio_ID
tp1 <- as.matrix(tp1[gids,,drop=FALSE]); storage.mode(tp1)<-"double"

megafit <- function(train_mat, pred_gids, secondary=NULL, k_factors, run_id){
  focal <- matrix(NA,length(gids),ncol(train_mat),dimnames=list(gids,colnames(train_mat)))
  focal[rownames(train_mat),] <- as.matrix(train_mat)
  Y <- focal; if(!is.null(secondary)) Y <- cbind(focal, secondary[gids,,drop=FALSE])
  rp <- MegaLMM_control(); rp$K<-k_factors; rp$h2_divisions<-20L; rp$burn<-0L; rp$thin<-THIN
  st <- setup_model_MegaLMM(Y=Y, formula=~(1|Line), data=lmm_data, relmat=list(Line=K), run_parameters=rp, run_ID=run_id)
  st <- set_priors_MegaLMM(st, MegaLMM_priors(tot_Y_var=list(V=0.5,nu=5), tot_F_var=list(V=18/20,nu=20),
        h2_priors_resids_fun=function(h2s,n)1, h2_priors_factors_fun=function(h2s,n)1,
        Lambda_prior=list(sampler=sample_Lambda_prec_ARD, Lambda_df=3, delta_1=list(shape=2,rate=1), delta_2=list(shape=3,rate=1), delta_iterations_factor=100)))
  st <- initialize_variables_MegaLMM(st); st <- fix_megalmm_posterior(st)
  maps <- make_Missing_data_map(st, max_NA_groups=Inf, verbose=FALSE); st <- set_Missing_data_map(st, maps$Missing_data_map)
  st <- initialize_MegaLMM(st, ncores=NCORES)
  for(b in 1:2){ st<-sample_MegaLMM(st,BURN); try(suppressWarnings(st<-reorder_factors(st,drop_cor_threshold=0.6)),silent=TRUE); st<-clear_Posterior(st); st<-fix_megalmm_posterior(st) }
  st <- sample_MegaLMM(st,N_ITER); st <- save_posterior_chunk(st)
  Eta <- load_posterior_param(st,"Eta_mean"); pred <- Eta[pred_gids, seq_len(ncol(focal)), drop=FALSE]
  rownames(pred)<-pred_gids; pred
}

kR <- function(nf) min(floor(length(gids)/4), floor(nf/2))
VARIANTS <- c("mega","tp1")
for(k in START_K:END_K){
  cat(sprintf("\n=== outer %d ===\n",k)); flush.console()
  fc <- cv[gids, k]; trn_gids <- gids[fc>=1 & fc<=5]; val_gids <- gids[fc==6]
  inner <- fc[match(trn_gids, gids)]
  # ---- scaler: ???????? x ???? ----
  tw <- data[data$bio_ID %in% trn_gids, ]
  mn <- sapply(tw[,trait_cols], min, na.rm=TRUE); mx <- sapply(tw[,trait_cols], max, na.rm=TRUE); rg <- pmax(mx-mn,1e-8)
  # ???? DMD ?? (???? + scale="none" + ?????)
  tw_n <- tw; tw_n[,trait_cols] <- as.data.frame(mapply(function(x,a,b)(x-a)/(b-a), tw[,trait_cols], mn, rg, SIMPLIFY=FALSE))
  comp <- find_DMD_components(tw_n, r=R_RANK, method="schur.DMD", scale="none")
  Rtr <- as.matrix(tensor_unroller(comp$R_all)); Ptr <- as.matrix(tensor_unroller(comp$phi_all))
  sv1 <- grep("SV1$", colnames(Ptr)); sv2 <- grep("SV2$", colnames(Ptr))
  # ???????? (??? scaler)
  vd <- data[data$bio_ID %in% val_gids, ]
  Xv <- array(NA, dim=c(length(trait_cols), length(DAS_ALL), length(val_gids)),
              dimnames=list(trait_cols, paste0("t",DAS_ALL), val_gids))
  for(g in val_gids){ sub <- vd[vd$bio_ID==g,]; sub <- sub[order(sub$DAS),]
    Xv[,,g] <- t(as.matrix(sweep(as.matrix(sub[,trait_cols]), 2, mn, "-") %*% diag(1/rg))) }
  # ???? CSV: ? = trait|DAS, ? = val_gids
  flat <- do.call(rbind, lapply(seq_along(trait_cols), function(pi) Xv[pi,,,drop=FALSE][1,,]))
  rownames(flat) <- as.vector(outer(trait_cols, paste0("t",DAS_ALL), paste, sep="|"))
  write.csv(flat, file.path(COMP, sprintf("iter%02d_val_X_norm.csv",k)))
  # ---- inner-5 ?? ----
  accR <- list(); accP <- list()
  for(v in VARIANTS){ accR[[v]] <- NULL; accP[[v]] <- NULL }
  for(j in 1:5){
    itr <- trn_gids[inner != j]; target <- c(trn_gids[inner == j], val_gids)
    rid <- function(m) sprintf("%s/run_%02d_inner%d_%s", OUTDIR, k, j, m)
    cat(sprintf("  inner %d (train %d) ...\n", j, length(itr))); flush.console()
    r0 <- megafit(Rtr[itr,,drop=FALSE], target, secondary=NULL, k_factors=kR(4), run_id=rid("mega_R"))
    q1 <- megafit(Ptr[itr,sv1,drop=FALSE], target, secondary=NULL, k_factors=kR(50), run_id=rid("mega_p1"))
    q2 <- megafit(Ptr[itr,sv2,drop=FALSE], target, secondary=NULL, k_factors=kR(50), run_id=rid("mega_p2"))
    pp <- matrix(NA,length(target),ncol(Ptr),dimnames=list(target,colnames(Ptr))); pp[,sv1]<-q1; pp[,sv2]<-q2
    accR[["mega"]] <- rbind(accR[["mega"]], r0[val_gids,,drop=FALSE]); accP[["mega"]] <- rbind(accP[["mega"]], pp[val_gids,,drop=FALSE])
    r1 <- megafit(Rtr[itr,,drop=FALSE], target, secondary=tp1, k_factors=kR(4+ncol(tp1)), run_id=rid("tp1_R"))
    s1 <- megafit(Ptr[itr,sv1,drop=FALSE], target, secondary=tp1, k_factors=kR(50+ncol(tp1)), run_id=rid("tp1_p1"))
    s2 <- megafit(Ptr[itr,sv2,drop=FALSE], target, secondary=tp1, k_factors=kR(50+ncol(tp1)), run_id=rid("tp1_p2"))
    qq <- matrix(NA,length(target),ncol(Ptr),dimnames=list(target,colnames(Ptr))); qq[,sv1]<-s1; qq[,sv2]<-s2
    accR[["tp1"]] <- rbind(accR[["tp1"]], r1[val_gids,,drop=FALSE]); accP[["tp1"]] <- rbind(accP[["tp1"]], qq[val_gids,,drop=FALSE])
  }
  nv <- length(val_gids)
  for(v in VARIANTS){
    # inner-5 集成: 每个 val 基因型在 5 个 inner fold 里各有一个预测 -> 按基因型平均
    VR <- matrix(0, nv, ncol(accR[[v]]), dimnames=list(val_gids, colnames(accR[[v]])))
    VP <- matrix(0, nv, ncol(accP[[v]]), dimnames=list(val_gids, colnames(accP[[v]])))
    for(j in 1:5){
      idx <- ((j-1)*nv + 1):(j*nv)
      VR <- VR + accR[[v]][idx,,drop=FALSE]
      VP <- VP + accP[[v]][idx,,drop=FALSE]
    }
    VR <- VR/5; VP <- VP/5
    write.csv(VR, file.path(COMP, sprintf("iter%02d_val_R_enspred_%s.csv", k, v)))
    write.csv(VP, file.path(COMP, sprintf("iter%02d_val_phi_enspred_%s.csv", k, v)))
  }
  cat(sprintf("  outer %d done (train=%d val=%d)\n", k, length(trn_gids), length(val_gids))); flush.console()
}
cat("DONE\n")
