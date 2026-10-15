#!/usr/bin/env Rscript
# build_cv.R -- 按当前品系集重划 10 折 CV
# 对齐作者做法: for i in 1:10 { set.seed(i); sample(1:6, n, replace=TRUE) }
library(data.table)
BASE <- "D:/pythonproject/knode/data/maize_RGB"
for (env in c("drought", "irrigated")) {
  d <- fread(file.path(BASE, sprintf("maize_rgb_%s_traits.csv", env)))
  ids <- sort(unique(d$bio_ID))
  n <- length(ids)
  cv <- matrix(NA_integer_, n, 10)
  for (i in 1:10) { set.seed(i); cv[, i] <- sample(1:6, n, replace = TRUE) }
  out <- data.frame(genotype = ids, cv, check.names = FALSE)
  colnames(out) <- c("genotype", paste0("V", 1:10))
  fwrite(out, file.path(BASE, sprintf("maize_rgb_%s_cv_10fold.csv", env)))
  cat(sprintf("[%s] n=%d -> maize_rgb_%s_cv_10fold.csv | fold1 dist: %s\n",
              env, n, env, paste(table(cv[, 1]), collapse="/")))
}