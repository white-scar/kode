#' find_DMD_components2.R ? ? find_DMD_components.R ??, ? gap ????"?????"
#' ????? DAS ? != 1 ???????? (???? gap ??? mask)
find_DMD_components <- function (data, r = 25, method = "calculated.A", scale = "min.max", time_gaps = 0) {
  data <- data[order(data$bio_ID, data$DAS), ]
  accessions <- unique(data$bio_ID)
  n_traits <- ncol(data) - 3
  n_timepoints <- length(unique(data$DAS))
  A_all <- array(NA, dim = c(n_traits, n_traits, length(accessions)), dimnames = list(colnames(data)[-c(1,2,3)], colnames(data)[-c(1,2,3)], accessions))
  X_all <- array(NA, dim = c(n_traits, n_timepoints, length(accessions)), dimnames = list(colnames(data)[-c(1,2,3)], paste0("t", unique(data$DAS)), accessions))
  if (method == "schur.DMD") {
    U_r_transpose_all <- array(NA, dim = c(r, n_traits, length(accessions)), dimnames = list(paste0("SV", 1:r), colnames(data)[-c(1,2,3)], accessions))
    sigma_r_all <- array(NA, dim = c(r, r, length(accessions)), dimnames = list(paste0("SV", 1:r), paste0("SV", 1:r), accessions))
    A_tilde_all <- array(NA, dim = c(r, r, length(accessions)), dimnames = list(paste0("SV", 1:r), paste0("SV", 1:r), accessions))
    Q_all <- array(NA, dim = c(r, r, length(accessions)), dimnames = list(paste0("SV", 1:r), paste0("SV", 1:r), accessions))
    R_all <- array(NA, dim = c(r, r, length(accessions)), dimnames = list(paste0("SV", 1:r), paste0("SV", 1:r), accessions))
    phi_all <- array(NA, dim = c(n_traits, r, length(accessions)), dimnames = list(colnames(data)[-c(1,2,3)], paste0("SV", 1:r), accessions))
  }
  if (scale == "min.max") {
    norm_minmax <- function(x) { (x - min(x)) / (max(x) - min(x)) }
    data[, -c(1:3)] <- as.data.frame(lapply(data[, -c(1:3)], norm_minmax))
  }
  das_seq <- sort(unique(data$DAS))
  drop_cols <- which(diff(das_seq) != 1)      # ?????????
  n_pairs <- (n_timepoints - 1) - length(drop_cols)   # ????????
  if (method == "schur.DMD")
    V_r_all <- array(NA, dim = c(n_pairs, r, length(accessions)), dimnames = list(paste0("t", 1:n_pairs), paste0("SV", 1:r), accessions))
  for (accession in accessions) {
    X <- t(data[which(data$bio_ID == accession), -c(1, 2, 3)])
    X_1 <- X[, -(ncol(X))]
    X_2 <- X[, -1]
    if (length(drop_cols) > 0) {            # ??? gap ???
      X_1 <- X_1[, -drop_cols, drop = FALSE]
      X_2 <- X_2[, -drop_cols, drop = FALSE]
    }
    if (method == "schur.DMD") {
      x <- svd(X_1)
      sigma <- matrix(0, ncol(x$u), ncol(x$u)); diag(sigma) <- x$d
      U <- x$u; V <- t(x$v)
      U_r_transpose <- Conj(t(U[, 1:r])); V_r <- t(V[1:r, ])
      sigma_r <- sigma[1:r, 1:r]; diag(sigma_r) <- 1 / diag(sigma_r); diag(sigma_r)[-c(1:r)] <- 0
      A_tilde <- U_r_transpose %*% X_2 %*% V_r %*% sigma_r
      schur <- Matrix::Schur(A_tilde, vectors = TRUE)
      Q <- schur$Q; R <- schur$T
      phi <- X_2 %*% V_r %*% sigma_r %*% Q
      A <- phi %*% R %*% pracma::pinv(phi)
      U_r_transpose_all[, , accession] <- U_r_transpose
      V_r_all[, , accession] <- V_r
      sigma_r_all[, , accession] <- sigma_r
      A_tilde_all[, , accession] <- A_tilde
      Q_all[, , accession] <- Q
      R_all[, , accession] <- R
      phi_all[, , accession] <- phi
    } else if (method == "calculated.A") {
      A <- X_2 %*% pracma::pinv(X_1)
    }
    A_all[, , accession] <- A
    X_all[, , accession] <- X
  }
  if (method == "schur.DMD") {
    return (list(A_all = A_all, X_all = X_all, U_r_transpose_all = U_r_transpose_all, V_r_all = V_r_all,
                 sigma_r_all = sigma_r_all, A_tilde_all = A_tilde_all, Q_all = Q_all, R_all = R_all, phi_all = phi_all))
  } else { return (list(A_all = A_all, X_all = X_all)) }
}
