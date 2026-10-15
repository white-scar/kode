# -*- coding: utf-8 -*-
# author_tp1_innerens_maize.R — 作者 tp1+MegaLMM+dynamicGP, 协议对齐 Kode: inner 5折 + 集成
# 每个 outer fold k: 5 个 inner 模型(各用训练集 4/5), 每个模型同时预测 (inner_val, outer_val);
#   CV 精度  = inner_val 的拼合预测 vs 真算子
#   验证精度 = 5 个模型对 outer_val 预测的平均 vs 真算子
args <- commandArgs(trailingOnly=TRUE)
START_K <- if(length(args)>=1) as.integer(args[1]) else 1
END_K   <- if(length(args)>=2) as.integer(args[2]) else 10
THIN    <- if(length(args)>=3) as.integer(args[3]) else 2
NCORES  <- if(length(args)>=4) as.integer(args[4]) else 8
OUTDIR  <- if(length(args)>=5) args[5] else "outputs/reproduce/author_tp1_maize_innerens"
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
cv   <- read.csv("data/maize_magic/cross_validation_folds_w_validation_set_maize.csv", row.names=1)
grm  <- as.matrix(read.csv("data/maize_magic/GRM_330x330_methodA_from552.csv", row.names=1, check.names=FALSE))
gids <- unique(data$bio_ID); grm <- grm[gids,gids]
K <- as.matrix(Matrix::forceSymmetric(grm)); colnames(K)<-rownames(K)<-gids
lmm_data <- data.frame(Line=gids, row.names=gids); trait_cols <- colnames(data)[-c(1,2,3)]
tp1_raw <- data[data$DAS==min(data$DAS),]; rownames(tp1_raw)<-tp1_raw$bio_ID
tp1 <- as.data.frame(lapply(tp1_raw[,trait_cols], function(x)(x-min(x))/(max(x)-min(x)))); rownames(tp1)<-tp1_raw$bio_ID
tp1 <- as.matrix(tp1[gids,,drop=FALSE]); storage.mode(tp1)<-"double"
# 一次拟合: 只观测 train_mat 的行, 其余为 NA -> 返回 pred_gids 的预测
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
  for(b in 1:2){ st <- sample_MegaLMM(st,BURN); try(suppressWarnings(st<-reorder_factors(st,drop_cor_threshold=0.6)),silent=TRUE); st<-clear_Posterior(st); st<-fix_megalmm_posterior(st) }
  st <- sample_MegaLMM(st,N_ITER); st <- save_posterior_chunk(st)
  Eta <- load_posterior_param(st,"Eta_mean")
  pred <- Eta[pred_gids, seq_len(ncol(focal)), drop=FALSE]; rownames(pred)<-pred_gids; pred
}
kR <- function(nf) min(floor(length(gids)/4), floor(nf/2))
res <- data.frame()
for(k in START_K:END_K){
  cat(sprintf("\n=== outer %d ===\n",k)); flush.console()
  Rtr <- as.matrix(read.csv(sprintf("outputs/reproduce/author_tp1_maize/components/iter%02d_trn_R.csv",k),row.names=1))
  Ptr <- as.matrix(read.csv(sprintf("outputs/reproduce/author_tp1_maize/components/iter%02d_trn_phi.csv",k),row.names=1))
  Rva <- as.matrix(read.csv(sprintf("outputs/reproduce/author_tp1_maize/components/iter%02d_val_R_true.csv",k),row.names=1))
  Pva <- as.matrix(read.csv(sprintf("outputs/reproduce/author_tp1_maize/components/iter%02d_val_phi_true.csv",k),row.names=1))
  trn_gids <- rownames(Rtr); val_gids <- rownames(Rva)
  inner <- cv[trn_gids, k]
  sv1<-grep("SV1$",colnames(Ptr)); sv2<-grep("SV2$",colnames(Ptr))
  CVr <- matrix(NA,length(trn_gids),ncol(Rtr),dimnames=list(trn_gids,colnames(Rtr)))
  CVp <- matrix(NA,length(trn_gids),ncol(Ptr),dimnames=list(trn_gids,colnames(Ptr)))
  VRl <- list(); VPl <- list()
  for(j in 1:5){
    itr <- which(inner!=j); iva <- trn_gids[inner==j]
    cat(sprintf("  inner %d (train %d, holdout %d) ...\n", j, length(itr), length(iva))); flush.console()
    target <- c(iva, val_gids)
    rid <- function(m) sprintf("%s/run_%02d_inner%d_%s", OUTDIR, k, j, m)
    pr <- megafit(Rtr[itr,,drop=FALSE], target, secondary=tp1, k_factors=kR(4+50), run_id=rid("R"))
    p1 <- megafit(Ptr[itr,sv1,drop=FALSE], target, secondary=tp1, k_factors=kR(50+50), run_id=rid("p1"))
    p2 <- megafit(Ptr[itr,sv2,drop=FALSE], target, secondary=tp1, k_factors=kR(50+50), run_id=rid("p2"))
    pp <- matrix(NA,length(target),ncol(Ptr),dimnames=list(target,colnames(Ptr)))
    pp[,sv1]<-p1; pp[,sv2]<-p2
    CVr[iva,] <- pr[iva,,drop=FALSE]; CVp[iva,] <- pp[iva,,drop=FALSE]
    VRl[[j]] <- pr[val_gids,,drop=FALSE]; VPl[[j]] <- pp[val_gids,,drop=FALSE]
  }
  acc <- function(pred,truth){ c<-intersect(rownames(pred),rownames(truth)); v<-sapply(seq_len(ncol(pred)),function(j){a<-truth[c,j];b<-pred[c,j];m<-is.finite(a)&is.finite(b); if(sum(m)>2&&sd(a[m])>1e-12&&sd(b[m])>1e-12) cor(a[m],b[m]) else NA_real_}); mean(v,na.rm=TRUE) }
  VRm <- Reduce(`+`, VRl)/length(VRl)
  VPm <- Reduce(`+`, VPl)/length(VPl)
  res <- rbind(res, data.frame(iter=k,
    R_cv=acc(CVr,Rtr), phi_cv=acc(CVp,Ptr),
    R_val=acc(VRm,Rva), phi_val=acc(VPm,Pva)))
  write.csv(CVr, file.path(COMP,sprintf("iter%02d_trn_R_cvpred.csv",k)))
  write.csv(CVp, file.path(COMP,sprintf("iter%02d_trn_phi_cvpred.csv",k)))
  write.csv(VRm, file.path(COMP,sprintf("iter%02d_val_R_enspred.csv",k)))
  write.csv(VPm, file.path(COMP,sprintf("iter%02d_val_phi_enspred.csv",k)))
  write.csv(res, file.path(OUTDIR,"author_tp1_innerens.csv"), row.names=FALSE)
  cat(sprintf("  outer %d: CV R=%.3f phi=%.3f | val R=%.3f phi=%.3f\n", k, res$R_cv[nrow(res)], res$phi_cv[nrow(res)], res$R_val[nrow(res)], res$phi_val[nrow(res)])); flush.console()
}
print(res); cat("DONE\n")
