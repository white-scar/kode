GCTA <- function (matrix_unrolled, kinship_filename = "GCTA_GRM/MAGIC.SNP.70k.impute.maf", wd = "/home/david/PycharmProjects/PhD_Projekt/Dynamic_Mode_Decomposition_Project/Maize_Data/SNP") {

  table <- cbind.data.frame(rownames(matrix_unrolled), rownames(matrix_unrolled), matrix_unrolled)
  trait_names <- colnames(matrix_unrolled)

  colnames(table) <- c("FAMIlY", "INDIV", paste0("Y", 1:(ncol(table) - 2)))
  rownames(table) <- NULL

  setwd(wd)

  data.table::fwrite(
    table,
    file = "temp_table.txt",
    sep = "\t",
    quote = F,
    eol = "\n",
    na = "\\N",
    qmethod = "escape"#,
  )

  h <- NULL

  dir.create("temp_files")

  for (i in seq_len(ncol(matrix_unrolled))) {
    system(paste0("./gcta-1.94.1 --grm ", kinship_filename, " --pheno temp_table.txt --mpheno ", i, " --reml --out temp_files/GREML_estimates_trait_", i, "_", trait_names[i]))
    if (file.exists(paste0("temp_files/GREML_estimates_trait_", i, "_", trait_names[i], ".hsq"))) {
      line <- readLines(paste0("temp_files/GREML_estimates_trait_", i, "_", trait_names[i], ".hsq"))
      h <- c(h, as.numeric(substr(line[5], 9, 16)))
    } else {
      h <- c(h, NA)
    }


  }

  file.remove("temp_table.txt")
  unlink("temp_files", recursive = TRUE)

  setwd("/home/david/PycharmProjects/PhD_Projekt/")

  names(h) <- trait_names

  return (h)
}