library(igraph)
set.seed(42)
BASE <- "D:/pythonproject/knode"
OUT  <- file.path(BASE, "reproduce/_mantel_export")
DEST <- file.path(BASE, "data/maize_RGB")
THR  <- 0.97
h2 <- read.csv(file.path(OUT, "h2.csv"), row.names = 1, check.names = FALSE)
allsel <- list()
for (env in c("drought", "irrigated")) {
  mtx <- as.matrix(read.csv(file.path(OUT, sprintf("manteln_%s.csv", env)), row.names = 1, check.names = FALSE))
  hcol <- paste0("h2_", env)
  mtx2 <- (mtx >= THR) * 1; diag(mtx2) <- 0
  g <- graph_from_adjacency_matrix(mtx2, mode = "max", diag = FALSE)
  cl <- cluster_louvain(g); mem <- membership(cl)
  rows <- data.frame()
  for (i in sort(unique(mem))) {
    tr <- names(mem)[mem == i]; vals <- h2[tr, hcol]
    pick <- if (all(is.na(vals))) tr[1] else tr[which.max(vals)]
    rows <- rbind(rows, data.frame(trait=pick, h2_mean=max(vals, na.rm=TRUE), cluster_size=length(tr), cluster=i))
  }
  rows <- rows[order(rows$trait), ]
  f <- file.path(DEST, sprintf("rgb_selected_traits_%s_thr097.csv", env))
  write.csv(rows, f, row.names = FALSE)
  cat(sprintf("[%s] thr=%.2f clusters=%d selected=%d -> %s\n", env, THR, length(unique(mem)), nrow(rows), basename(f)))
  cat("   ", paste(sort(rows$trait), collapse=", "), "\n\n")
  allsel[[env]] <- sort(rows$trait)
}
cat(sprintf("重叠 %d 个\n", length(intersect(allsel$drought, allsel$irrigated))))
cat("  drought only  :", paste(setdiff(allsel$drought, allsel$irrigated), collapse=", "), "\n")
cat("  irrigated only:", paste(setdiff(allsel$irrigated, allsel$drought), collapse=", "), "\n")