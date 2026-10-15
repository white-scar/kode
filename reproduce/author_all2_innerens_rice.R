# -*- coding: utf-8 -*-
# author_all2_innerens_rice.R -- 作者 fig2 链路第 2 段: mega / tp1 两个变体的 inner-5 集成 (rice)
#   组件来源: outputs/reproduce/author_tp1_rice/components/ (由 _author_tp1_export_rice.R 生成)
#   与 dynamicGP_MegaLMM_TP1_*.R 的区别: 那里是单次拟合; 本脚本 inner-5 集成, 与 Kode 对齐
#   注: 作者 fig2 里 rice 只有 mega/tp1 两条 (rr 变体另跑且不需要), 故本脚本不含 rr/snps
#   输出: <OUTDIR>/components/iter%02d_val_{R,phi}_enspred_<variant>.csv  + author_all2_innerens.csv
args <- commandArgs(trailingOnly=TRUE)
K0 <- if(length(args)>=1) as.integer(args[1]) else 1
K1 <- if(length(args)>=2) as.integer(args[2]) else 10
THIN <- if(length(args)>=3) as.integer(args[3]) else 2
NCORES <- if(length(args)>=4) as.integer(args[4]) else 8
OUTDIR <- if(length(args)>=5) args[5] else "outputs/reproduce/author_all2_rice_innerens"
COMPIN <- if(length(args)>=6) args[6] else "outputs/reproduce/author_tp1_rice/components"
ROOT <- local({
  fa <- grep("^--file=", commandArgs(FALSE), value=TRUE)
  start <- if (length(fa)) dirname(normalizePath(sub("^--file=", "", fa[1]))) else getwd()
  p <- start; for (i in 1:8) { if (dir.exists(file.path(p,"data")) && dir.exists(file.path(p,"outputs"))) break; p <- dirname(p) }
  p
})
setwd(ROOT)
COMP <- file.path(OUTDIR, "components"); dir.create(COMP, recursive=TRUE, showWarnings=FALSE)
for (f in c("table_to_tensor.R","A_r_from_phi_and_R.R")) source(file.path("reproduce/R", f))
suppressPackageStartupMessages(library(MegaLMM))
Sys.setenv(OMP_NUM_THREADS=as.character(NCORES))
try(suppressWarnings(MegaLMM::set_MegaLMM_nthreads(NCORES)), silent=TRUE)
try(suppressWarnings(MegaLMM::set_omp_nthreads(NCORES)), silent=TRUE)
R_RANK <- 2; BURN <- 200; N_ITER <- 300
fix_megalmm_posterior <- function(st){ for(p in st$Posterior$posteriorSample_params){ if(is.null(st$Posterior[[p]])&&!is.null(st$current_state[[p]])&&length(dim(st$current_state[[p]]))==2){ st$Posterior[[p]]<-array(0,dim=c(0,dim(st$current_state[[p]]))); dimnames(st$Posterior[[p]])[2:3]<-dimnames(st$current_state[[p]])}}; for(p in st$Posterior$posteriorMean_params){ if(is.null(st$Posterior[[p]])&&!is.null(st$current_state[[p]])&&length(dim(st$current_state[[p]]))==2){ st$Posterior[[p]]<-array(0,dim=dim(st$current_state[[p]])); dimnames(st$Posterior[[p]])<-dimnames(st$current_state[[p]])}}; st }

data <- read.csv("data/rice_salinity/rice_indica_control_traits.csv")[, -1]
data <- data[data$DAS >= 30, ]
data$bio_ID <- as.character(data$bio_ID)
cv <- read.csv("data/rice_salinity/rice_cv_10fold.csv", row.names=1); rownames(cv) <- as.character(rownames(cv))
grm <- as.matrix(read.csv("data/rice_salinity/rice_GRM_226x226.csv", row.names=1, check.names=FALSE))
gids <- unique(data$bio_ID); grm <- grm[gids, gids]
K <- as.matrix(Matrix::forceSymmetric(grm)); colnames(K) <- rownames(K) <- gids
lmm_data <- data.frame(Line=gids, row.names=gids); trait_cols <- colnames(data)[-c(1,2,3)]
tp1_raw <- data[data$DAS==30, ]; rownames(tp1_raw) <- tp1_raw$bio_ID
tp1 <- as.data.frame(lapply(tp1_raw[,trait_cols], function(x)(x-min(x))/(max(x)-min(x)))); rownames(tp1) <- tp1_raw$bio_ID
tp1 <- as.matrix(tp1[gids,,drop=FALSE]); storage.mode(tp1) <- "double"

megafit <- function(train_mat, pred_gids, secondary=NULL, k_factors, run_id){
  focal <- matrix(NA,length(gids),ncol(train_mat),dimnames=list(gids,colnames(train_mat)))
  focal[rownames(train_mat),] <- as.matrix(train_mat)
  Y <- focal; if(!is.null(secondary)) Y <- cbind(focal, secondary[gids,,drop=FALSE])
  rp <- MegaLMM_control(); rp$K<-k_factors; rp$h2_divisions<-20L; rp$burn<-0L; rp$thin<-THIN
  st <- setup_model_MegaLMM(Y=Y, formula=~(1|Line), data=lmm_data, relmat=list(Line=K), run_parameters=rp, run_ID=run_id)
  st <- set_priors_MegaLMM(st, MegaLMM_priors(tot_Y_var=list(V=0.5,nu=5), tot_F_var=list(V=18/20,nu=20),
        h2_priors_resids_fun=function(h2s,n)1, h2_priors_factors_fun=function(h2s,n)1,
        Lambda_prior=list(sampler=sample_Lambda_prec_ARD, Lambda_df=3, delta_1=list(shape=2,rate=1), delta_2=list(shape=3,rate=1))))
  st <- initialize_variables_MegaLMM(st); st <- fix_megalmm_posterior(st)
  maps <- make_Missing_data_map(st, max_NA_groups=Inf, verbose=FALSE); st <- set_Missing_data_map(st, maps$Missing_data_map)
  st <- initialize_MegaLMM(st, ncores=NCORES)
  for(b in 1:2){ st <- sample_MegaLMM(st,BURN); try(suppressWarnings(st<-reorder_factors(st,drop_cor_threshold=0.6)),silent=TRUE) }
  st <- sample_MegaLMM(st,N_ITER); st <- save_posterior_chunk(st)
  Eta <- load_posterior_param(st,"Eta_mean")
  pred <- Eta[pred_gids, seq_len(ncol(focal)), drop=FALSE]; rownames(pred)<-pred_gids; pred
}

kR <- function(nf) min(floor(length(gids)/4), floor(nf/2))
VARIANTS <- c("mega","tp1")
res <- data.frame()
for(k in K0:K1){
  cat(sprintf("\n=== outer %d ===\n",k)); flush.console()
  Rtr <- as.matrix(read.csv(sprintf("%s/iter%02d_trn_R.csv", COMPIN, k), row.names=1))
  Ptr <- as.matrix(read.csv(sprintf("%s/iter%02d_trn_phi.csv", COMPIN, k), row.names=1))
  Rva <- as.matrix(read.csv(sprintf("%s/iter%02d_val_R_true.csv", COMPIN, k), row.names=1))
  Pva <- as.matrix(read.csv(sprintf("%s/iter%02d_val_phi_true.csv", COMPIN, k), row.names=1))
  trn_gids <- rownames(Rtr); val_gids <- rownames(Rva)
  inner <- cv[trn_gids, k]
  sv1 <- grep("SV1$",colnames(Ptr)); sv2 <- grep("SV2$",colnames(Ptr))
  CVr <- list(); CVp <- list(); VR <- list(); VP <- list()
  for(v in VARIANTS){ CVr[[v]] <- matrix(NA,length(trn_gids),ncol(Rtr),dimnames=list(trn_gids,colnames(Rtr)))
                      CVp[[v]] <- matrix(NA,length(trn_gids),ncol(Ptr),dimnames=list(trn_gids,colnames(Ptr)))
                      VR[[v]] <- list(); VP[[v]] <- list() }
  for(j in 1:5){
    itr <- which(inner!=j); iva <- trn_gids[inner==j]; target <- c(iva, val_gids)
    rid <- function(m) sprintf("%s/run_%02d_inner%d_%s", OUTDIR, k, j, m)
    cat(sprintf("  inner %d: mega ...\n", j)); flush.console()
    prm <- megafit(Rtr[itr,,drop=FALSE], target, secondary=NULL, k_factors=kR(4), run_id=rid("mega_R"))
    pm1 <- megafit(Ptr[itr,sv1,drop=FALSE], target, secondary=NULL, k_factors=kR(50), run_id=rid("mega_p1"))
    pm2 <- megafit(Ptr[itr,sv2,drop=FALSE], target, secondary=NULL, k_factors=kR(50), run_id=rid("mega_p2"))
    ppm <- matrix(NA,length(target),ncol(Ptr),dimnames=list(target,colnames(Ptr))); ppm[,sv1]<-pm1; ppm[,sv2]<-pm2
    CVr[["mega"]][iva,] <- prm[iva,,drop=FALSE]; CVp[["mega"]][iva,] <- ppm[iva,,drop=FALSE]
    VR[["mega"]] <- c(VR[["mega"]], list(prm[val_gids,,drop=FALSE])); VP[["mega"]] <- c(VP[["mega"]], list(ppm[val_gids,,drop=FALSE]))
    cat(sprintf("  inner %d: tp1 ...\n", j)); flush.console()
    prt <- megafit(Rtr[itr,,drop=FALSE], target, secondary=tp1, k_factors=kR(4+50), run_id=rid("tp1_R"))
    pt1 <- megafit(Ptr[itr,sv1,drop=FALSE], target, secondary=tp1, k_factors=kR(50+50), run_id=rid("tp1_p1"))
    pt2 <- megafit(Ptr[itr,sv2,drop=FALSE], target, secondary=tp1, k_factors=kR(50+50), run_id=rid("tp1_p2"))
    ppt <- matrix(NA,length(target),ncol(Ptr),dimnames=list(target,colnames(Ptr))); ppt[,sv1]<-pt1; ppt[,sv2]<-pt2
    CVr[["tp1"]][iva,] <- prt[iva,,drop=FALSE]; CVp[["tp1"]][iva,] <- ppt[iva,,drop=FALSE]
    VR[["tp1"]] <- c(VR[["tp1"]], list(prt[val_gids,,drop=FALSE])); VP[["tp1"]] <- c(VP[["tp1"]], list(ppt[val_gids,,drop=FALSE]))
  }
  acc <- function(pred,truth){ c<-intersect(rownames(pred),rownames(truth)); v<-sapply(seq_len(ncol(pred)),function(j){a<-truth[c,j]; b<-pred[c,j]; ok<-is.finite(a)&is.finite(b); if(sum(ok)<3||sd(a[ok])<1e-12||sd(b[ok])<1e-12) NA_real_ else cor(a[ok],b[ok])}); mean(v,na.rm=TRUE)}
  for(v in VARIANTS){
    # 5 个 inner 的验证预测逐基因型平均 (修 2026-09-20: 原来是 rbind 后对全部行均值, 只剩 1 行 -> rownames 报错)
    VRm <- Reduce(`+`, VR[[v]])/length(VR[[v]])
    VPm <- Reduce(`+`, VP[[v]])/length(VP[[v]])
    rownames(VRm)<-val_gids; rownames(VPm)<-val_gids
    res <- rbind(res, data.frame(iter=k, variant=v,
      R_cv=acc(CVr[[v]],Rtr), phi_cv=acc(CVp[[v]],Ptr),
      R_val=acc(VRm,Rva), phi_val=acc(VPm,Pva)))
    write.csv(VRm, file.path(COMP, sprintf("iter%02d_val_R_enspred_%s.csv", k, v)))
    write.csv(VPm, file.path(COMP, sprintf("iter%02d_val_phi_enspred_%s.csv", k, v)))
  }
  write.csv(res, file.path(OUTDIR,"author_all2_innerens.csv"), row.names=FALSE)
  cat(sprintf("  outer %d done\n", k)); flush.console()
}
print(res); cat("DONE\n")