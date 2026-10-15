# author_all3_innerens_maize.R
# 作者三个变体 (mega / tp1 / rr) 的 inner-5 集成, 训练折归一化
#   组件来源: outputs/reproduce/author_tp1_maize/components/  (由 author_tp1_export_maize.R 用训练折 scaler + scale="none" 生成)
#   与 dynamicGP_MegaLMM_TP1_maize.R 的区别: 那里是单次拟合; 本脚本是 inner-5 集成, 与 Kode 对齐
# 输出: <OUTDIR>/components/iterXX_val_{R,phi}_enspred_<variant>.csv
args <- commandArgs(trailingOnly=TRUE)
START_K <- if(length(args)>=1) as.integer(args[1]) else 1
END_K   <- if(length(args)>=2) as.integer(args[2]) else 10
THIN    <- if(length(args)>=3) as.integer(args[3]) else 2
NCORES  <- if(length(args)>=4) as.integer(args[4]) else 8
OUTDIR  <- if(length(args)>=5) args[5] else "outputs/reproduce/author_all3_maize_innerens"
COMP    <- file.path(OUTDIR,"components"); dir.create(COMP, recursive=TRUE, showWarnings=FALSE)
setwd("/data/fandata/dashuaige")
for (f in c("find_DMD_components.R","tensor_unroller.R","table_to_tensor.R",
            "A_r_from_phi_and_R.R","all_timepoints_predictor.R"))
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
snps <- readRDS("data/maize_magic/snps.rds")
snps <- as.matrix(snps[gids, ]); storage.mode(snps) <- "double"

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

rrblup_predict <- function(train_mats, val_gids, cvcol){
  trn_gids <- rownames(train_mats[[1]])
  all_Y <- do.call(cbind, train_mats); n_el <- ncol(all_Y)
  fold_acc <- matrix(NA, n_el, 5); store <- vector("list", n_el)
  for(i in seq_len(n_el)){
    store[[i]] <- vector("list", 5); y <- all_Y[,i]; names(y) <- trn_gids
    for(j in 1:5){
      tr <- trn_gids[which(cvcol[trn_gids,2] != j)]; te <- trn_gids[which(cvcol[trn_gids,2] == j)]
      m <- rrBLUP::mixed.solve(y=y[tr], Z=snps[tr,,drop=FALSE])
      me <- as.numeric(m$u); b <- as.numeric(m$beta)
      fold_acc[i,j] <- cor(as.numeric(snps[te,,drop=FALSE] %*% me) + b, y[te], use="complete.obs")
      store[[i]][[j]] <- list(BLUE=b, me=me)
    }
  }
  best_fold <- which.max(colMeans(fold_acc, na.rm=TRUE))
  out <- matrix(NA, length(val_gids), n_el, dimnames=list(val_gids, colnames(all_Y)))
  for(i in seq_len(n_el)){ ss <- store[[i]][[best_fold]]; out[,i] <- as.numeric(snps[val_gids,,drop=FALSE] %*% ss$me) + ss$BLUE }
  out
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
  cvcol <- cv[, c(1, k+1)]; rownames(cvcol) <- cvcol[,1]
  inner <- cv[trn_gids, k]
  sv1 <- grep("SV1$",colnames(Ptr)); sv2 <- grep("SV2$",colnames(Ptr))
  VARIANTS <- c("mega","tp1","rr")
  CVr <- list(); CVp <- list(); VR <- list(); VP <- list()
  for(v in VARIANTS){ CVr[[v]] <- matrix(NA,length(trn_gids),ncol(Rtr),dimnames=list(trn_gids,colnames(Rtr)))
                       CVp[[v]] <- matrix(NA,length(trn_gids),ncol(Ptr),dimnames=list(trn_gids,colnames(Ptr)))
                       VR[[v]] <- NULL; VP[[v]] <- NULL }
  for(j in 1:5){
    itr <- which(inner!=j); iva <- trn_gids[inner==j]; target <- c(iva, val_gids)
    rid <- function(m) sprintf("%s/run_%02d_inner%d_%s", OUTDIR, k, j, m)
    # ---- mega (no secondary) ----
    cat(sprintf("  inner %d: mega ...\n", j)); flush.console()
    prm <- megafit(Rtr[itr,,drop=FALSE], target, secondary=NULL, k_factors=kR(4), run_id=rid("mega_R"))
    pm1 <- megafit(Ptr[itr,sv1,drop=FALSE], target, secondary=NULL, k_factors=kR(50), run_id=rid("mega_p1"))
    pm2 <- megafit(Ptr[itr,sv2,drop=FALSE], target, secondary=NULL, k_factors=kR(50), run_id=rid("mega_p2"))
    ppm <- matrix(NA,length(target),ncol(Ptr),dimnames=list(target,colnames(Ptr))); ppm[,sv1]<-pm1; ppm[,sv2]<-pm2
    CVr[["mega"]][iva,] <- prm[iva,,drop=FALSE]; CVp[["mega"]][iva,] <- ppm[iva,,drop=FALSE]
    VR[["mega"]] <- rbind(VR[["mega"]], prm[val_gids,,drop=FALSE]); VP[["mega"]] <- rbind(VP[["mega"]], ppm[val_gids,,drop=FALSE])
    # ---- tp1 (secondary = all t1 traits) ----
    cat(sprintf("  inner %d: tp1 ...\n", j)); flush.console()
    prt <- megafit(Rtr[itr,,drop=FALSE], target, secondary=tp1, k_factors=kR(4+50), run_id=rid("tp1_R"))
    pt1 <- megafit(Ptr[itr,sv1,drop=FALSE], target, secondary=tp1, k_factors=kR(50+50), run_id=rid("tp1_p1"))
    pt2 <- megafit(Ptr[itr,sv2,drop=FALSE], target, secondary=tp1, k_factors=kR(50+50), run_id=rid("tp1_p2"))
    ppt <- matrix(NA,length(target),ncol(Ptr),dimnames=list(target,colnames(Ptr))); ppt[,sv1]<-pt1; ppt[,sv2]<-pt2
    CVr[["tp1"]][iva,] <- prt[iva,,drop=FALSE]; CVp[["tp1"]][iva,] <- ppt[iva,,drop=FALSE]
    VR[["tp1"]] <- rbind(VR[["tp1"]], prt[val_gids,,drop=FALSE]); VP[["tp1"]] <- rbind(VP[["tp1"]], ppt[val_gids,,drop=FALSE])
    # ---- rr (rrBLUP on SNPs) ----
    cat(sprintf("  inner %d: rr ...\n", j)); flush.console()
    prr <- rrblup_predict(list(R=Rtr[itr,,drop=FALSE], phi=Ptr[itr,,drop=FALSE]), target, cvcol)
    Rrr <- prr[, seq_len(ncol(Rtr)), drop=FALSE]; Prr <- prr[, ncol(Rtr)+seq_len(ncol(Ptr)), drop=FALSE]
    CVr[["rr"]][iva,] <- Rrr[iva,,drop=FALSE]; CVp[["rr"]][iva,] <- Prr[iva,,drop=FALSE]
    VR[["rr"]] <- rbind(VR[["rr"]], Rrr[val_gids,,drop=FALSE]); VP[["rr"]] <- rbind(VP[["rr"]], Prr[val_gids,,drop=FALSE])
  }
  acc <- function(pred,truth){ c<-intersect(rownames(pred),rownames(truth)); v<-sapply(seq_len(ncol(pred)),function(j){a<-truth[c,j];b<-pred[c,j];m<-is.finite(a)&is.finite(b); if(sum(m)>2&&sd(a[m])>1e-12&&sd(b[m])>1e-12) cor(a[m],b[m]) else NA_real_}); mean(v,na.rm=TRUE) }
  for(v in VARIANTS){
    VRm <- Reduce(`+`, lapply(seq_len(nrow(VR[[v]])), function(i) VR[[v]][i,,drop=FALSE]))/nrow(VR[[v]])
    VPm <- Reduce(`+`, lapply(seq_len(nrow(VP[[v]])), function(i) VP[[v]][i,,drop=FALSE]))/nrow(VP[[v]])
    rownames(VRm)<-val_gids; rownames(VPm)<-val_gids
    res <- rbind(res, data.frame(iter=k, variant=v,
      R_cv=acc(CVr[[v]],Rtr), phi_cv=acc(CVp[[v]],Ptr),
      R_val=acc(VRm,Rva), phi_val=acc(VPm,Pva)))
    write.csv(VRm, file.path(COMP, sprintf("iter%02d_val_R_enspred_%s.csv", k, v)))
    write.csv(VPm, file.path(COMP, sprintf("iter%02d_val_phi_enspred_%s.csv", k, v)))
  }
  write.csv(res, file.path(OUTDIR,"author_all3_innerens.csv"), row.names=FALSE)
  cat(sprintf("  outer %d done\n", k)); flush.console()
}
print(res); cat("DONE\n")
