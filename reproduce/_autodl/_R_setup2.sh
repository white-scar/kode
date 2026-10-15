#!/bin/bash
# _R_setup2.sh -- 补装: libgsl-dev + 升级 Matrix + lme4/RcppZiggurat + MegaLMM
set -u
Q=/root/autodl-tmp/qkn
LOG=$Q/_R_setup2.log
: > $LOG
export DEBIAN_FRONTEND=noninteractive
echo "[$(date '+%F %T')] apt install libgsl-dev" >> $LOG
apt-get install -y libgsl-dev >> $LOG 2>&1
cat > $Q/_R_pkgs2.R <<'EOF'
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"), Ncpus = 16)
need <- c("Matrix","lme4","RcppZiggurat","RcppGSL")
for (p in need) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
cat("--- versions ---\n"); for (p in need) cat(p, as.character(packageVersion(p)), "\n")
remotes::install_github("deruncie/MegaLMM", upgrade = "never", Ncpus = 16)
cat("--- check ---\n")
print(requireNamespace("MegaLMM", quietly = TRUE))
library(MegaLMM); cat("MegaLMM loaded OK, version", as.character(packageVersion("MegaLMM")), "\n")
EOF
echo "[$(date '+%F %T')] R packages round 2" >> $LOG
Rscript $Q/_R_pkgs2.R >> $LOG 2>&1
rc=$?
echo "[$(date '+%F %T')] DONE rc=$rc" >> $LOG
date '+%F %T' > $Q/_R_setup2.DONE