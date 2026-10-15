# -*- coding: utf-8 -*-
#' h2_by_flight.R -- UAV-RGB 逐 flight 的 SNP-based heritability (GREML)
#'
#' 用途: 为 build_traits.py 的时间点质控提供依据 (见该脚本 timepoint_qc()).
#'   step 1  每个 flight: 逐性状 mixed.solve(y, K) 估 h2 = Vu/(Vu+Ve), 再对 34 个 RGB VI 取平均
#'   step 2  drought 的这套 h2 用于时间点筛选 (irrigated 的时间点集与 drought 保持一致)
#'   step 3  剔除规则: (i) 缺失率 >50% → DAP27/34; (ii) 平均 h2 < 0.12 → DAP110/127
#'   最终保留 12 个观测时间点: 42,56,61,64,68,71,75,78,82,84,90,99
#'
#' 输入 : data/maize_RGB/YYCD18_RGB.csv, YYCI18_RGB.csv, geno_GRM.csv
#' 输出 : data/maize_RGB/maize_rgb_h2_by_flight.csv   (DAP, h2_drought, h2_irrigated, h2_mean, subset)
#' 用法 : Rscript data/maize_RGB/h2_by_flight.R        (从仓库根目录运行; 约 10-18 min)
library(rrBLUP)

ROOT <- local({
  fa <- grep("^--file=", commandArgs(FALSE), value=TRUE)
  p <- if (length(fa)) dirname(normalizePath(sub("^--file=", "", fa[1]))) else getwd()
  for (i in 1:8) { if (dir.exists(file.path(p, "data")) && dir.exists(file.path(p, "reproduce"))) break; p <- dirname(p) }
  p
})
setwd(ROOT)

FLIGHTS <- c(27, 34, 42, 56, 61, 64, 68, 71, 75, 78, 82, 84, 90, 99, 110, 127)
THR_H2  <- 0.12        # 平均 h2 阈值
grm <- as.matrix(read.csv("data/maize_RGB/geno_GRM.csv", row.names = 1, check.names = FALSE))

h2_one <- function(y, K) {
  y <- as.numeric(y); ok <- is.finite(y)
  if (sum(ok) < 20 || sd(y[ok]) < 1e-12) return(NA_real_)
  m <- try(rrBLUP::mixed.solve(y = y[ok], K = K[ok, ok, drop = FALSE]), silent = TRUE)
  if (inherits(m, "try-error")) return(NA_real_)
  as.numeric(m$Vu / (m$Vu + m$Ve))
}

h2_per_flight <- function(csv, tcol) {
  d <- read.csv(csv, check.names = FALSE); d$Entry <- as.character(d$Entry)
  vi <- setdiff(colnames(d)[5:41], c("Red", "Green", "Blue"))   # 34 个 RGB VI
  keep <- intersect(unique(d$Entry), rownames(grm))
  K <- grm[keep, keep, drop = FALSE]
  miss <- sapply(FLIGHTS, function(fl) {                        # 缺失率 (Entry x flight)
    sub <- d[d[[tcol]] == fl & d$Entry %in% keep, c("Entry", vi)]
    mean(is.na(sub[, vi])) })
  h2 <- sapply(FLIGHTS, function(fl) {
    sub <- d[d[[tcol]] == fl & d$Entry %in% keep, ]
    hs <- sapply(vi, function(t) {
      x <- sub[, c("Entry", t)]; x <- x[is.finite(x[[t]]), ]
      if (!nrow(x)) return(NA_real_)
      y <- tapply(x[[t]], x$Entry, mean)
      h2_one(y, K[names(y), names(y), drop = FALSE]) })
    mean(hs, na.rm = TRUE) })
  list(h2 = h2, missing = miss)
}

dr <- h2_per_flight("data/maize_RGB/YYCD18_RGB.csv", "Flight")
ir <- h2_per_flight("data/maize_RGB/YYCI18_RGB.csv", "DAP")

subset_tag <- ifelse(is.na(dr$h2), "bare",
              ifelse(dr$missing > 0.5, "dropped_missing",
              ifelse(dr$h2 < THR_H2, "dropped_low-h2", "kept")))
out <- data.frame(DAP = FLIGHTS, h2_drought = dr$h2, h2_irrigated = ir$h2,
                  h2_mean = (dr$h2 + ir$h2) / 2,
                  missing_drought = dr$missing, subset = subset_tag)
write.csv(out, "data/maize_RGB/maize_rgb_h2_by_flight.csv", row.names = FALSE)
cat("saved data/maize_RGB/maize_rgb_h2_by_flight.csv\n")
print(out, row.names = FALSE, digits = 4)
cat("\nkept flights:", paste(out$DAP[out$subset == "kept"], collapse = ","), "\n")