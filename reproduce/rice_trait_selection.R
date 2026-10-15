# ============================================================================
# rice_trait_selection.R
# Replicate the author's (Hobby et al.) trait clustering + selection logic on the
# rice indica-control dataset:
#   1) per trait x DAS heritability  (GREML via rrBLUP::mixed.solve with the GRM)
#   2) pairwise inter-trait Mantel correlation of the spatio-temporal X matrices
#      (same as trait_clustering_and_selection.R: dist(X_all[i,,]))
#   3) threshold the Mantel matrix -> graph clustering -> keep, within each
#      cluster, the trait with the highest mean heritability
#   4) write the filtered phenotype table
# igraph/ade4 are unavailable on this server -> clustering via (a) connected
# components at threshold t and (b) hclust cut at height 1-t; both are reported.
# ============================================================================
setwd("/data/fandata/dashuaige")
suppressPackageStartupMessages(library(rrBLUP))

ph  <- read.csv("data/rice_salinity/rice_indica_control_traits.csv", row.names=1)
grm <- as.matrix(read.csv("data/rice_salinity/rice_GRM_226x226.csv", row.names=1, check.names=FALSE))
ph$bio_ID <- as.character(ph$bio_ID)
rownames(grm) <- colnames(grm) <- as.character(rownames(grm))
gids <- intersect(unique(ph$bio_ID), rownames(grm))
cat(sprintf("genotypes=%d  traits=%d  DAS=%d\n", length(gids), ncol(ph)-3, length(unique(ph$DAS))))
tc <- colnames(ph)[-c(1,2,3)]
DAS <- sort(unique(ph$DAS)); T <- length(DAS); P <- length(tc); N <- length(gids)
G <- grm[gids, gids]

# ---- phenotype tensor (min-max per trait, as in find_DMD_components default) ----
X <- array(NA, dim=c(P, T, N), dimnames=list(tc, paste0("t",DAS), gids))
for (pi in seq_along(tc)) {
  sub <- ph[ph$bio_ID %in% gids, c("bio_ID","DAS",tc[pi])]
  m <- matrix(NA, N, T, dimnames=list(gids, DAS))
  for (i in seq_len(nrow(sub))) m[as.character(sub$bio_ID[i]), as.character(sub$DAS[i])] <- sub[i, tc[pi]]
  rng <- diff(range(m, na.rm=TRUE))
  m <- (m - min(m, na.rm=TRUE)) / ifelse(rng > 1e-12, rng, 1)
  X[pi,,] <- t(m)                       # (T x N)
}

# ---- 1) heritability per trait x DAS, then averaged per trait ----
h_rows <- list()
for (pi in seq_along(tc)) for (ti in seq_len(T)) {
  y <- X[pi, ti, ]; names(y) <- gids
  fit <- try(rrBLUP::mixed.solve(y = y, K = G), silent = TRUE)
  if (inherits(fit, "try-error")) next
  Vu <- as.numeric(fit$Vu); Ve <- as.numeric(fit$Ve)
  h_rows[[length(h_rows)+1]] <- data.frame(Trait=tc[pi], DAS=DAS[ti], H2=Vu/(Vu+Ve), Vu=Vu, Ve=Ve)
}
h <- do.call(rbind, h_rows)
write.csv(h, "outputs/reproduce/rice_trait_selection_heritability_by_time.csv", row.names=FALSE)
h_agg <- aggregate(H2 ~ Trait, data=h, FUN=mean)
h_agg$SD <- aggregate(H2 ~ Trait, data=h, FUN=sd)$H2
h_agg <- h_agg[order(-h_agg$H2), ]
write.csv(h_agg, "outputs/reproduce/rice_trait_selection_heritability.csv", row.names=FALSE)
cat("\n=== mean heritability per trait (sorted) ===\n"); print(h_agg, row.names=FALSE)

# ---- 2) pairwise inter-trait Mantel correlation ----
D <- lapply(seq_along(tc), function(pi) as.matrix(dist(X[pi,,])))
mant <- matrix(1, P, P, dimnames=list(tc, tc))
for (i in 1:P) for (j in 1:P) if (i != j)
  mant[i,j] <- cor(D[[i]][upper.tri(D[[i]])], D[[j]][upper.tri(D[[j]])])
write.csv(mant, "outputs/reproduce/rice_trait_selection_mantel.csv")
cat("\n=== Mantel correlation: quantiles ===\n")
ut <- mant[upper.tri(mant)]
print(round(quantile(ut, c(0,.25,.5,.75,.9,.95,1)), 4))
cat("pairs >=0.99:", sum(ut>=0.99), " >=0.97:", sum(ut>=0.97), " >=0.96:", sum(ut>=0.96), " >=0.90:", sum(ut>=0.90), " of", length(ut), "\n")

# ---- 3) clustering across a threshold grid ----
cat("\n=== clustering at various thresholds ===\n")
cat(sprintf("%6s %8s %8s %10s\n", "thr", "n_cc", "n_hcl", "kept"))
for (t in c(0.90, 0.95, 0.96, 0.97, 0.98, 0.99)) {
  A <- (mant >= t); diag(A) <- FALSE
  # connected components (base R)
  comp <- integer(P); names(comp) <- tc; k <- 0
  for (i in 1:P) if (comp[i]==0) { k <- k+1; q <- i; comp[i] <- k
    while (length(q)) { v <- q[1]; q <- q[-1]
      nb <- which(A[v,] & comp==0); if (length(nb)) { comp[nb] <- k; q <- c(q, nb) } } }
  ncc <- length(unique(comp))
  hc <- cutree(hclust(as.dist(1 - mant), method="average"), h = 1 - t)
  nhc <- length(unique(hc))
  # selection under the connected-component clustering
  sel <- unlist(lapply(unique(comp), function(g) {
    cl <- names(comp)[comp==g]; cl[which.max(h_agg$H2[match(cl, h_agg$Trait)])] }))
  cat(sprintf("%6.2f %8d %8d %10d\n", t, ncc, nhc, length(sel)))
  if (t == 0.96) {
    sel_cc <- sel; comp_cc <- comp
    sel_hc <- unlist(lapply(unique(hc), function(g) {
      cl <- names(hc)[hc==g]; cl[which.max(h_agg$H2[match(cl, h_agg$Trait)])] }))
    write.csv(data.frame(Trait=tc, cc_cluster=comp_cc, hclust_cluster=hc[as.character(tc)],
                         H2=h_agg$H2[match(tc, h_agg$Trait)]),
              "outputs/reproduce/rice_trait_selection_clusters.csv", row.names=FALSE)
    write.csv(data.frame(selected_cc=sel_cc), "outputs/reproduce/rice_trait_selection_selected_cc096.csv", row.names=FALSE)
    write.csv(data.frame(selected_hclust=sel_hc), "outputs/reproduce/rice_trait_selection_selected_hclust096.csv", row.names=FALSE)
    cat("\n--- clusters @0.96 (connected components) ---\n")
    for (g in unique(comp_cc)) cat(sprintf("  C%d: %s\n", g, paste(sort(names(comp_cc)[comp_cc==g]), collapse=", ")))
    cat("\nselected @0.96 (cc):  ", paste(sort(sel_cc), collapse=", "), "\n")
    cat("selected @0.96 (hcl): ", paste(sort(sel_hc), collapse=", "), "\n")
    # write filtered dataset (cc-based selection)
    keep <- c("bio_ID","accession","DAS", intersect(sort(sel_cc), colnames(ph)))
    out <- ph[, keep]
    write.csv(out, "data/rice_salinity/rice_indica_control_traits_selected.csv")
    cat(sprintf("\nwrote filtered dataset: %d traits -> %d traits\n", P, length(keep)-3))
  }
}
