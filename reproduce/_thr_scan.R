library(igraph)
set.seed(42)
BASE <- "D:/pythonproject/knode"
OUT  <- file.path(BASE, "reproduce/_mantel_export")
for (env in c("drought", "irrigated")) {
  mtx <- as.matrix(read.csv(file.path(OUT, sprintf("manteln_%s.csv", env)), row.names = 1, check.names = FALSE))
  cat(sprintf("=== %s ===\n", env))
  cat("  thr   clusters  within  out    ratio  diff\n")
  for (THR in seq(0.90, 0.995, 0.005)) {
    mtx2 <- (mtx >= THR) * 1; diag(mtx2) <- 0
    g <- graph_from_adjacency_matrix(mtx2, mode = "max", diag = FALSE)
    cl <- cluster_louvain(g); mem <- membership(cl)
    w <- c(); o <- c()
    for (j in sort(unique(mem))) {
      idx <- which(mem == j)
      w <- c(w, mean(mtx[idx, idx]))
      o <- c(o, mean(mtx[idx, -idx]))
    }
    wm <- mean(w, na.rm = TRUE); om <- mean(o, na.rm = TRUE)
    cat(sprintf("  %.3f  %3d      %.3f  %.3f  %.2f   %.3f\n", THR, length(unique(mem)), wm, om, wm/om, wm-om))
  }
  cat("\n")
}