# -*- coding: utf-8 -*-
# ?? tp1+MegaLMM+dynamicGP ??(CV3)????? (rec_t1 / rec_tk / iter)
# ????: (1) inner-5 ??  (2) scaler = ????? x DAS<=40 ? per-trait min/max???????
#           (3) ????????
# ?? dynamicGP_MegaLMM_TP1_maize_CV3_t1.R ????? tk ?? + NMSE/R2?
args <- commandArgs(trailingOnly=TRUE)
START_K <- if(length(args)>=1) as.integer(args[1]) else 1
END_K   <- if(length(args)>=2) as.integer(args[2]) else 10
THIN    <- if(length(args)>=3) as.integer(args[3]) else 2
NCORES  <- if(length(args)>=4) as.integer(args[4]) else 16
OUTDIR  <- if(length(args)>=5) args[5] else "outputs/reproduce/dynamicGP_MegaLMM_maize_CV3_3scen"
dir.create(OUTDIR, recursive=TRUE, showWarnings=FALSE)
setwd("/data/fandata/dashuaige")
for (f in c("find_DMD_components.R","tensor_unroller.R","table_to_tensor.R","A_r_from_phi_and_R.R","all_timepoints_predictor.R"))
  source(file.path("reproduce/R", f))
suppressPackageStartupMessages(library(MegaLMM))
R_RANK <- 2; BURN <- 200; N_ITER <- 300; TRLEN <- 20   # TRLEN = DAS40 ? vtimes ????
Sys.setenv(OMP_NUM_THREADS=as.character(NCORES))
try(suppressWarnings(MegaLMM::set_MegaLMM_nthreads(NCORES)), silent=TRUE); try(suppressWarnings(MegaLMM::set_omp_nthreads(NCORES)), silent=TRUE)
fix_megalmm_posterior <- function(st){ for(p in st$Posterior$posteriorSample_params){ if(is.null(st$Posterior[[p]])&&!is.null(st$current_state[[p]])&&length(dim(st$current_state[[p]]))==2){ st$Posterior[[p]]<-array(0,dim=c(0,dim(st$current_state[[p]]))); dimnames(st$Posterior[[p]])[2:3]<-dimnames(st$current_state[[p]])}}; for(p in st$Posterior$posteriorMean_params){ if(is.null(st$Posterior[[p]])&&!is.null(st$current_state[[p]])&&length(dim(st$current_state[[p]]))==2){ st$Posterior[[p]]<-array(0,dim=dim(st$current_state[[p]])); dimnames(st$Posterior[[p]])<-dimnames(st$current_state[[p]])}}; st }
data <- read.csv("data/maize_magic/maize_trait_BLUP_50_NArmv_traits_afterClustering.csv")[,-1]
cv   <- read.csv("data/maize_magic/cross_validation_folds_w_validation_set_maize.csv", row.names=1)
grm  <- as.matrix(read.csv("data/maize_magic/GRM_330x330_methodA_from552.csv", row.names=1, check.names=FALSE))
gids <- unique(data$bio_ID); grm <- grm[gids,gids]
K <- as.matrix(Matrix::forceSymmetric(grm)); colnames(K)<-rownames(K)<-gids
lmm_data <- data.frame(Line=gids, row.names=gids); trait_cols <- colnames(data)[-c(1,2,3)]
tp1_raw <- data[data$DAS==15,]; rownames(tp1_raw)<-tp1_raw$bio_ID
tp1 <- as.data.frame(lapply(tp1_raw[,trait_cols], function(x)(x-min(x))/(max(x)-min(x)))); rownames(tp1)<-tp1_raw$bio_ID
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
predict_dynamics <- function(A_all, X_all, TRLEN=20){
  lines <- intersect(dimnames(A_all)[[3]], dimnames(X_all)[[3]]); traits <- dimnames(A_all)[[1]]; n_tp <- dim(X_all)[2]
  mk <- function() array(NA, dim=c(length(traits),n_tp,length(lines)), dimnames=list(traits,paste0("t",1:n_tp),lines))
  mat <- mat_rec <- mat_tk <- mk()
  for(ln in lines){ X<-X_all[,,ln]; A<-A_all[,,ln]
    m<-m_rec<-m_tk<-matrix(0,nrow(X),n_tp); m[,1]<-m_rec[,1]<-m_tk[,1]<-X[,1]
    for(t in 1:(n_tp-1)){
      m[,t+1]     <- Re(A%*%X[,t])
      m_rec[,t+1] <- Re(A%*%m_rec[,t])
      if(t < TRLEN)       m_tk[,t+1] <- Re(A%*%m_tk[,t])   # ??? t_k
      else if(t == TRLEN) m_tk[,t+1] <- Re(A%*%X[,TRLEN])  # t_k ????
      else                m_tk[,t+1] <- Re(A%*%m_tk[,t])   # ????
    }
    mat[,,ln]<-m; mat_rec[,,ln]<-m_rec; mat_tk[,,ln]<-m_tk }
  metrics <- function(P){ tr<-rep(traits,each=n_tp); tm<-rep(seq_len(n_tp),times=length(traits))
    PCC<-MSE<-NMSE<-R2<-numeric(length(tr))
    for(i in seq_along(tr)){ y<-X_all[tr[i],tm[i],]; p<-P[tr[i],tm[i],]; ok<-is.finite(y)&is.finite(p)
      if(sum(ok)<3){ PCC[i]<-MSE[i]<-NMSE[i]<-R2[i]<-NA_real_; next }
      y<-y[ok]; p<-p[ok]; mse<-mean((y-p)^2); vy<-var(y)
      MSE[i]<-mse; NMSE[i]<-if(vy>0) mse/vy else NA_real_; R2[i]<-if(vy>0) 1-mse/vy else NA_real_; PCC[i]<-cor(y,p) }
    data.frame(Trait=tr,Time=paste0("t",tm),PCC=PCC,MSE=MSE,NMSE=NMSE,R2=R2,stringsAsFactors=FALSE) }
  list(iter=metrics(mat), rec=metrics(mat_rec), tk=metrics(mat_tk))
}
all_rows <- list()
for(k in START_K:END_K){
  cat(sprintf("\n=== outer %d ===\n",k)); flush.console()
  fc <- cv[gids,k]; trn_gids <- gids[fc!=6]; val_gids <- gids[fc==6]
  tw <- data[data$bio_ID %in% trn_gids & data$DAS<=40, ]
  comp <- find_DMD_components(tw, r=R_RANK, method="schur.DMD", time_gaps=3)
  Rtr <- as.matrix(tensor_unroller(comp$R_all)); Ptr <- as.matrix(tensor_unroller(comp$phi_all))
  sv1<-grep("SV1$",colnames(Ptr)); sv2<-grep("SV2$",colnames(Ptr))
  mn <- sapply(tw[,trait_cols], min, na.rm=TRUE); mx <- sapply(tw[,trait_cols], max, na.rm=TRUE); rg <- pmax(mx-mn,1e-8)
  vd <- data[data$bio_ID %in% val_gids & data$DAS>=15, ]
  vtimes <- sort(unique(vd$DAS))
  Xv <- array(NA, dim=c(length(trait_cols), length(vtimes), length(val_gids)), dimnames=list(trait_cols, paste0("t",vtimes), val_gids))
  for(g in val_gids){ sub <- vd[vd$bio_ID==g,]; sub <- sub[order(sub$DAS),]
    Xv[,,g] <- t(as.matrix(sweep(as.matrix(sub[,trait_cols]), 2, mn, "-") %*% diag(1/rg))) }
  kR <- function(nf) min(floor(length(gids)/4), floor(nf/2))
  inner <- fc[fc!=6]
  Rp_list<-list(); Pp_list<-list()
  for(j in 1:5){
    itr <- trn_gids[inner!=j]
    cat(sprintf("  inner %d (train %d) ...\n", j, length(itr))); flush.console()
    r1 <- megafit(Rtr[itr,,drop=FALSE], val_gids, secondary=tp1, k_factors=kR(4+50), run_id=sprintf("%s/run_%02d_i%d_R",OUTDIR,k,j))
    q1 <- megafit(Ptr[itr,sv1,drop=FALSE], val_gids, secondary=tp1, k_factors=kR(50+50), run_id=sprintf("%s/run_%02d_i%d_p1",OUTDIR,k,j))
    q2 <- megafit(Ptr[itr,sv2,drop=FALSE], val_gids, secondary=tp1, k_factors=kR(50+50), run_id=sprintf("%s/run_%02d_i%d_p2",OUTDIR,k,j))
    pp <- matrix(NA,length(val_gids),ncol(Ptr),dimnames=list(val_gids,colnames(Ptr))); pp[,sv1]<-q1; pp[,sv2]<-q2
    Rp_list[[j]]<-r1; Pp_list[[j]]<-pp
  }
  Rp <- Reduce(`+`,Rp_list)/length(Rp_list); Pp <- Reduce(`+`,Pp_list)/length(Pp_list)
  Rv <- table_to_tensor(as.data.frame(Rp), component="R", r=R_RANK); PHIv <- table_to_tensor(as.data.frame(Pp), component="phi", r=R_RANK)
  Av <- A_r_from_phi_and_R(PHIv, Rv); m <- predict_dynamics(Av, Xv, TRLEN)
  for(cfg in c("iter","rec","tk")){ d<-m[[cfg]]; d$Iteration<-k; d$Method<-paste0("tp1_",cfg); all_rows[[paste(k,cfg)]]<-d }
  cat(sprintf("  outer %d done\n", k)); flush.console()
}
tab <- do.call(rbind, all_rows)
write.csv(tab, file.path(OUTDIR,"snapshot_metrics_long.csv"), row.names=FALSE)
print(aggregate(PCC ~ Method + Time, data=tab, FUN=function(x) mean(x, na.rm=TRUE)))
cat("DONE rows:", nrow(tab), "\n")
