#!/bin/bash
# _R_setup.sh -- 在 autodl 上装 R + MegaLMM (CPU MCMC 包), 供作者的 dynamicGP 复现脚本用
set -u
Q=/root/autodl-tmp/qkn
LOG=$Q/_R_setup.log
: > $LOG
echo "[$(date '+%F %T')] apt-get update" >> $LOG
export DEBIAN_FRONTEND=noninteractive
apt-get update -y >> $LOG 2>&1
echo "[$(date '+%F %T')] apt-get install r-base / build deps" >> $LOG
apt-get install -y r-base r-base-dev gfortran libcurl4-openssl-dev libssl-dev libxml2-dev >> $LOG 2>&1
echo "[$(date '+%F %T')] R version:" >> $LOG
R --version >> $LOG 2>&1
cat > $Q/_R_pkgs.R <<'EOF'
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"), Ncpus = 16)
pk <- c("Rcpp","RcppEigen","Matrix","RhpcBLASctl","coda","MASS","roptim","numDeriv","remotes")
for (p in pk) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
stopifnot(requireNamespace("remotes", quietly = TRUE))
remotes::install_github("deruncie/MegaLMM", upgrade = "never", Ncpus = 16)
cat("--- check ---\n")
print(requireNamespace("MegaLMM", quietly = TRUE))
library(MegaLMM); cat("MegaLMM loaded OK\n")
EOF
echo "[$(date '+%F %T')] install R packages (CRAN + GitHub MegaLMM)" >> $LOG
Rscript $Q/_R_pkgs.R >> $LOG 2>&1
echo "[$(date '+%F %T')] DONE rc=$?" >> $LOG
date '+%F %T' > $Q/_R_setup.DONE