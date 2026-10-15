#!/bin/bash
# _R_setup3.sh -- 升级 Matrix (源码, 锁 1.6-5) -> lme4 -> MegaLMM
set -u
Q=/root/autodl-tmp/qkn
LOG=$Q/_R_setup3.log
: > $LOG
cat > $Q/_R_pkgs3.R <<'EOF'
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"), Ncpus = 16)
cat("当前 Matrix:", as.character(packageVersion("Matrix")), "\n")
ok <- FALSE
for (v in c("1.6-5", "1.5-4.1")) {
  cat("尝试安装 Matrix", v, "...\n")
  r <- try(remotes::install_version("Matrix", version = v, upgrade = "never", Ncpus = 16), silent = TRUE)
  if (!inherits(r, "try-error") && as.character(packageVersion("Matrix")) == v) { ok <- TRUE; break }
}
cat("Matrix ->", as.character(packageVersion("Matrix")), "ok=", ok, "\n")
install.packages("lme4", Ncpus = 16)
cat("lme4:", requireNamespace("lme4", quietly = TRUE), "\n")
remotes::install_github("deruncie/MegaLMM", upgrade = "never", Ncpus = 16)
cat("MegaLMM installed:", requireNamespace("MegaLMM", quietly = TRUE), "\n")
library(MegaLMM); cat("MegaLMM loaded OK version", as.character(packageVersion("MegaLMM")), "\n")
EOF
echo "[$(date '+%F %T')] round 3 start" >> $LOG
Rscript $Q/_R_pkgs3.R >> $LOG 2>&1
rc=$?
echo "[$(date '+%F %T')] DONE rc=$rc" >> $LOG
date '+%F %T' > $Q/_R_setup3.DONE