options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"), Ncpus = 16)
for (p in c("pracma", "corpcor", "stringr", "MASS", "reshape2")) {
  if (!requireNamespace(p, quietly = TRUE)) { cat("installing", p, "\n"); install.packages(p) }
}
for (p in c("pracma", "corpcor", "stringr", "MASS", "reshape2", "MegaLMM")) {
  cat(sprintf("%-12s %s\n", p, if (requireNamespace(p, quietly = TRUE)) as.character(packageVersion(p)) else "MISSING"))
}