# -*- coding: utf-8 -*-
# author_tp1_innercv_maize.R -- 作者 tp1+MegaLMM 的 inner 5-fold CV 精度 (训练集 cross-validation)
args <- commandArgs(trailingOnly=TRUE)
START_K <- if(length(args)>=1) as.integer(args[1]) else 1
END_K   <- if(length(args)>=2) as.integer(args[2]) else 10
THIN    <- if(length(args)>=3) as.integer(args[3]) else 2
NCORES  <- if(length(args)>=4) as.integer(args[4]) else 8
OUTDIR  <- if(length(args)>=5) args[5] else "outputs/reproduce/author_tp1_maize"
COMP    <- file.path(OUTDIR,"components"); dir.create(COMP, recursive=TRUE, showWarnings=FALSE)
setwd("/data/fandata/dashuaige")
for (f in c("find_DMD_components.R","tensor_unroller.R","table_to_tensor.R","A_r_from_phi_and_R.R","all_timepoints_predictor.R"))
  source(file.path("reproduce/R", f))
suppressPackageStartupMessages(library(MegaLMM)); suppressPackageStartupMessages(library(rrBLUP))
R_RANK <- 2; BURN <- 200; N_ITER <- 300
Sys.setenv(OMP_NUM_THREADS=as.character(NCORES))
try(suppressWarnings(MegaLMM::set_MegaLMM_nthreads(NCORES)), silent=TRUE)
try(suppressWarnings(MegaLMM::set_omp_nthreads(NCORES)), silent=TRUE)
fix_megalmm_posterior <- function(st){ for(p in st$Posterior$posteriorSample_params){ if(is.null(st$Posterior[[p]])&&!is.null(st$current_state[[p]])&&length(dim(st$current_state[[p]]))==2){ st$Posterior[[p]]<-array(0,dim=c(0,dim(st$current_state[[p]]))); dimnames(st$Posterior[[p]])[2:3]<-dimnames(st$current_state[[p]])}}; for(p in st$Posterior$posteriorMean_params){ if(is.null(st$Posterior[[p]])&&!is.null(st$current_state[[p]])&&length(dim(st$current_state[[p]]))==2){ st$Posterior[[p]]<-array(0,dim=dim(st$current_state[[p]])); dimnames(st$Posterior[[p]])<-dimnames(st$current_state[[p]])}}; st }
data <- read.csv("data/maize_magic/maize_trait_BLUP_50_NArmv_traits_afterClustering.csv")[,-1]
cv   <- read.csv("data/maize_magic/cross_validation_folds_w_validation_set_maize.csv")
grm  <- as.matrix(read.csv("data/maize_magic/GRM_330x330_methodA_from552.csv", row.names=1, check.names=FALSE))
gids <- unique(data$bio_ID); grm <- grm[gids,gids]
K <- as.matrix(Matrix::forceSymmetric(grm)); colnames(K)<-rownames(K)<-gids
lmm_data <- data.frame(Line=gids, row.names=gids); trait_cols <- colnames(data)[-c(1,2,3)]
tp1_raw <- data[data$DAS==min(data$DAS),]; rownames(tp1_raw)<-tp1_raw$bio_ID
tp1 <- as.data.frame(lapply(tp1_raw[,trait_cols], function(x)(x-min(x))/(max(x)-min(x)))); rownames(tp1)<-tp1_raw$bio_ID
tp1 <- as.matrix(tp1[gids,,drop=FALSE]); storage.mode(tp1)<-"double"

megafit <- function(train_mat, pred_gids, secondary=NULL, k_factors, run_id, mask=TRUE){
  focal <- matrix(NA,length(gids),ncol(train_mat),dimnames=list(gids,colnames(train_mat)))
  focal[rownames(train_mat),] <- as.matrix(train_mat)
  Y <- focal; if(!is.null(secondary)) Y <- cbind(focal, secondary[gids,,drop=FALSE])
  Y2 <- Y; if(mask) Y2[pred_gids, seq_len(ncol(focal))] <- NA
  rp <- MegaLMM_control(); rp$K<-k_factors; rp$h2_divisions<-20L; rp$burn<-0L; rp$thin<-THIN
  st <- setup_model_MegaLMM(Y=Y2, formula=~(1|Line), data=lmm_data, relmat=list(Line=K), run_parameters=rp, run_ID=run_id)
  st <- set_priors_MegaLMM(st, MegaLMM_priors(tot_Y_var=list(V=0.5,nu=5), tot_F_var=list(V=18/20,nu=20),
        h2_priors_resids_fun=function(h2s,n)1, h2_priors_factors_fun=function(h2s,n)1,
        Lambda_prior=list(sampler=sample_Lambda_prec_ARD, Lambda_df=3, delta_1=list(shape=2,rate=1), delta_2=list(shape=3,rate=1), delta_iterations_factor=100)))
  st <- initialize_variables_MegaLMM(st); st <- fix_megalmm_posterior(st)
  maps <- make_Missing_data_map(st, max_NA_groups=Inf, verbose=FALSE); st <- set_Missing_data_map(st, maps$Missing_data_map)
  st <- initialize_MegaLMM(st, ncores=NCORES)
  for(b in 1:2){ st <- sample_MegaLMM(st,BURN); try(suppressWarnings(st<-reorder_factors(st,drop_cor_threshold=0.6)),silent=TRUE); st<-clear_Posterior(st); st<-fix_megalmm_posterior(st) }
  st <- sample_MegaLMM(st,N_ITER); st <- save_posterior_chunk(st)
  Eta <- load_posterior_param(st,"Eta_mean"); pred <- Eta[pred_gids, seq_len(ncol(focal)), drop=FALSE]
  rownames(pred)<-pred_gids; pred
}
acc <- function(pred, truth){ c<-intersect(rownames(pred),rownames(truth)); mean(sapply(seq_len(ncol(pred)), function(j) cor(truth[c,j],pred[c,j],use="complete.obs")), na.rm=TRUE) }
kR <- function(nf) min(floor(length(gids)/4), floor(nf/2))

res <- data.frame()
for(k in START_K:END_K){
  cat(sprintf("\n=== outer %d ===\n",k)); flush.console()
  trn_R <- as.matrix(read.csv(file.path(COMP,sprintf("iter%02d_trn_R.csv",k)),row.names=1))
  trn_P <- as.matrix(read.csv(file.path(COMP,sprintf("iter%02d_trn_phi.csv",k)),row.names=1))
  trn_gids <- rownames(trn_R)
  inner <- cv[trn_gids, k+1]                      # 1..5 inner train folds (6=outer val)
  sv1<-grep("SV1$",colnames(trn_P)); sv2<-grep("SV2$",colnames(trn_P))
  P_R <- matrix(NA,length(trn_gids),ncol(trn_R),dimnames=list(trn_gids,colnames(trn_R)))
  P_P <- matrix(NA,length(trn_gids),ncol(trn_P),dimnames=list(trn_gids,colnames(trn_P)))
  for(j in 1:5){
    itr <- which(inner!=j); ipr <- which(inner==j)
    d <- file.path(OUTDIR,sprintf("run_%02d_inner%d",k,j)); 
    cat(sprintf("  inner %d (train %d / pred %d) ...\n", j, length(itr), length(ipr))); flush.console()
    P_R[ipr,] <- megafit(trn_R[itr,,drop=FALSE], trn_gids[ipr], secondary=tp1, k_factors=kR(4+50), run_id=paste0(d,"_R"), mask=TRUE)
    p1 <- megafit(trn_P[itr,sv1,drop=FALSE], trn_gids[ipr], secondary=tp1, k_factors=kR(50+50), run_id=paste0(d,"_p1"), mask=TRUE)
    p2 <- megafit(trn_P[itr,sv2,drop=FALSE], trn_gids[ipr], secondary=tp1, k_factors=kR(50+50), run_id=paste0(d,"_p2"), mask=TRUE)
    P_P[ipr,sv1] <- p1; P_P[ipr,sv2] <- p2
  }
  aR <- acc(P_R, trn_R); aP <- acc(P_P, trn_P)
  res <- rbind(res, data.frame(iter=k, R_cv=aR, phi_cv=aP))
  write.csv(P_R, file.path(COMP,sprintf("iter%02d_trn_R_cvpred.csv",k)))
  write.csv(P_P, file.path(COMP,sprintf("iter%02d_trn_phi_cvpred.csv",k)))
  write.csv(res, file.path(OUTDIR,"author_tp1_innercv.csv"), row.names=FALSE)
  cat(sprintf("  outer %d: R_cv=%.3f phi_cv=%.3f\n", k, aR, aP)); flush.console()
}
print(res); cat("DONE\n")
