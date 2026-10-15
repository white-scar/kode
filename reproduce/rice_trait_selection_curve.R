# threshold curve (mirrors the author's trait_clustering_and_selection.R grid)
# choose the threshold that maximises mean-within / mean-out cluster Mantel correlation
setwd("/data/fandata/dashuaige")
suppressPackageStartupMessages(library(rrBLUP))
mant <- as.matrix(read.csv("outputs/reproduce/rice_trait_selection_mantel.csv", row.names=1))
h <- read.csv("outputs/reproduce/rice_trait_selection_heritability.csv")
tc <- rownames(mant); P <- length(tc)

cluster_cc <- function(A){ comp <- integer(P); k <- 0
  for (i in 1:P) if (comp[i]==0) { k <- k+1; q <- i; comp[i] <- k
    while (length(q)) { v <- q[1]; q <- q[-1]; nb <- which(A[v,] & comp==0)
      if (length(nb)) { comp[nb] <- k; q <- c(q, nb) } } }
  comp }

res <- list()
for (t in seq(0.80, 1.0, by=0.005)) {
  A <- (mant >= t); diag(A) <- FALSE
  comp <- cluster_cc(A); hc <- cutree(hclust(as.dist(1-mant), method="average"), h=1-t)
  for (kind in c("cc","hcl")) {
    lab <- if (kind=="cc") comp else hc
    gs <- unique(lab); wi <- numeric(length(gs)); ou <- numeric(length(gs))
    for (g in gs) { in_ <- which(lab==g); out <- setdiff(seq_len(P), in_)
      if (length(in_) > 1) { sb <- mant[in_, in_]; wi[g] <- mean(sb[upper.tri(sb)]) } else wi[g] <- NA
      ou[g] <- mean(mant[in_, out]) }
    mw <- mean(wi, na.rm=TRUE); mo <- mean(ou, na.rm=TRUE)
    sel <- unlist(lapply(gs, function(g){ cl <- tc[lab==g]; cl[which.max(h$H2[match(cl,h$Trait)])] }))
    res[[length(res)+1]] <- data.frame(kind=kind, thr=t, n_clust=length(gs), within=mw, out=mo,
                                       ratio=mw/mo, diff=mw-mo, n_selected=length(sel))
  }
}
R <- do.call(rbind, res)
write.csv(R, "outputs/reproduce/rice_trait_selection_threshold_curve.csv", row.names=FALSE)
cat("=== best thresholds by within/out ratio ===\n")
for (k in c("cc","hcl")) {
  s <- R[R$kind==k, ]; b <- s[which.max(s$ratio), ]
  cat(sprintf("%s : best thr=%.3f  n_clust=%d  within=%.3f out=%.3f ratio=%.2f  kept=%d traits\n",
              k, b$thr, b$n_clust, b$within, b$out, b$ratio, b$n_selected))
  print(head(s[order(-s$ratio), c("thr","n_clust","within","out","ratio","n_selected")], 6), row.names=FALSE)
}
# detailed selection for a few candidate thresholds (cc)
for (t in c(0.95, 0.96, 0.97, 0.98, 0.99)) {
  A <- (mant >= t); diag(A) <- FALSE; comp <- cluster_cc(A)
  sel <- unlist(lapply(unique(comp), function(g){ cl <- tc[comp==g]; cl[which.max(h$H2[match(cl,h$Trait)])] }))
  cat(sprintf("\nthr=%.2f  cc-clusters=%d  selected(%d): %s\n", t, length(unique(comp)), length(sel), paste(sort(sel), collapse=", ")))
  for (g in unique(comp)) cat(sprintf("   C%d(%d): %s\n", g, sum(comp==g), paste(tc[comp==g], collapse=", ")))
}
