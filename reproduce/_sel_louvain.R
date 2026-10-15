library(igraph)
set.seed(42)
BASE <- "D:/pythonproject/knode"
OUT  <- file.path(BASE, "reproduce/_mantel_export")
h2 <- read.csv(file.path(OUT, "h2.csv"), row.names = 1, check.names = FALSE)
for (env in c("drought", "irrigated")) {
  mtx <- as.matrix(read.csv(file.path(OUT, sprintf("manteln_%s.csv", env)), row.names = 1, check.names = FALSE))
  hcol <- paste0("h2_", env)
  for (THR in c(0.96, 0.97)) {
    mtx2 <- mtx
    mtx2[mtx2 >= THR] <- 1; mtx2[mtx2 < THR] <- 0
    diag(mtx2) <- 0
    g <- graph_from_adjacency_matrix(mtx2, mode = "max", diag = FALSE)
    cl <- cluster_louvain(g); mem <- membership(cl)
    sel <- c()
    for (i in sort(unique(mem))) {
      tr <- names(mem)[mem == i]; vals <- h2[tr, hcol]
      sel <- c(sel, if (all(is.na(vals))) tr[1] else tr[which.max(vals)])
    }
    cat(sprintf("[%s thr=%.2f] clusters=%d selected=%d\n", env, THR, length(unique(mem)), length(sel)))
    cat("   ", paste(sort(sel), collapse=", "), "\n")
  }
  cat("\n")
}