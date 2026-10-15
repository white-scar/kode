--- page 1 ---
Data and script files for the genome wide association studies for the 2015 salt
experiments on the indica and aus Rice panels reported in Al-Tamimi et al.
(2016)
H. Oakey, N. Al-Tamimi1, B. Berger, C. J. Brien, S. Saade, Y. S. Ho, S. M. Schm¨ockel, M.
Tester2 and S. Negra˜o
1email: nadia.tamimi@kaust.edu.sa
2email: Mark.Tester@kaust.edu.sa
August 3, 2016
--- page 2 ---
Thisdocumentdescribesthedataandcodefilesthatwereusedinthegenomewideassociationstudy(GWAS)
phase for the two experiments reported by Al-Tamimi et al. (2016): experiment 0254 that investigates an indica
panelandexperiment0255thatinvestigatesanaus panel. Inaddition,thecombinedsinglesetpanels(INDAUS)
was subject to a GWAS. These files have been provided in GWAS data and scripts.zip.
The Phenotypic data are the BLUEs produced in the phenotypic analysis phase by fitting Mixed Lin-
ear Models (MLM) to the phenotpic responses. This data and the scripts to produce it are in Phenotyp-
ing data and script files.zip. Genotypic data were composed of the HDRA dataset, a 700K SNP array designed
by McCouchs laboratory at Cornell University (McCouch, 2016).
We used a GWAS approach referred to as the interaction model, amalgamates each phenotypic response for
both treatments (salt and control). This approach uses fits an MLM that is fitted to includes the kinship (K)
matrix as a random effect to control for population structure (Yang et al., 2014; Chen et al., 2011). When
analyzing the combined population INDAUS, we also included principal components to control for population
structure, because of the deep population structure in rice (Zhao et al., 2014; Yu et al., 2006).
1 Files in GWAS data and scripts.zip
1.1 Files in the directory Cleaned SNP and kinship data
This folder contains the cleaned SNP data where only only SNP markers with a minor allele frequency of
≥ 0.05 and a number of accessions with a minor allele > 6 were used for association analyses. SNP data
is in the Hapmap format, while kinship and principal component files were produced using TASSEL software
(http://www.maizegenetics.net/tassel).
(cid:136) Hapmap AUS700.maf005.nomissing.hmp.txt – txt files containing the cleaned SNP data for aus panel.
(cid:136) Hapmap IND700.maf005.nomissing.hmp – txt files containing the cleaned SNP data for indica panel.
(cid:136) Hapmap INDAUS700.maf005.nomissing.hmp – txt files containing the cleaned SNP data for INDAUS
panel.
(cid:136) Kin AUS700.maf005.nomissing – txt files containing the kinship data for aus panel.
(cid:136) Kin IND700.maf005.nomissing – txt files containing the kinship data for indica panel.
(cid:136) Kin INDAUS700.maf005.nomissing – txt files containing the kinship data for INDAUS panel.
(cid:136) PC INDAUS700 – txt files containing the principle component data.
1.2 Files in the directory ASReml interaction R scripts
We developed a GWAS interaction model to account for the marker, and the interaction of the marker (SNP)
and treatment (control and salt). The inclusion of the Marker-by-Treatment term, allows identifying markers
that have a significant contribution depending on the treatment. The scripts contain code to be executed using
the R statistical computing environment (R Development Core Team, 2015).
Data processing R scripts: 1. readindata aus.R – an R script to import the aus txt files containing the
raw data for cleaned SNP and kinship and amends files in the write format for the interaction model.
2. readindata ind.R – an R script to import the indica txt files containing the raw data for cleaned SNP
and kinship and amends files in the write format for the interaction model.
3. readindata ausindica pca.R – an R script to import the INDAUS txt files containing the raw data for
cleanedSNP,kinshipandprinciplecomponentandamendsfilesinthewriteformatfortheinteraction
model.
GWAS analysis R scripts: TheRpackageASReml-R(Butleretal.,2009)isusedtoperformthemixedmodel
analysis for the GWAS, qvalue (Dabney and Warnes, 2004) is used to performs Bonferroni and false
discovery rate (FDR) statistical tests and qqman (Turner, 2014) is used to produce the Manhattan plots.
1
--- page 3 ---
1. analysedata aus.R – an R script that uses the package ASReml-R to run the GWAS interaction model
on the aus data without principle component analysis (PCA).
2. analysedata ind.R – an R script that uses the R package ASReml-R to run the GWAS interaction
model on the indica data without principle component analysis (PCA).
3. analysedata ausind. R – an R script that uses the package ASReml-R to run the GWAS interaction
model on the INDAUS data with principle component analysis (PCA).
4. map.R — an R script that adds the chromosome and base pairs columns into the output of analyse-
data.R in the right order.
5. Manplots.R — an R script that uses the R package qvalue to perform Bonferroni and false discovery
rate (FDR) statistical tests and uses the R package qqman to plot the Manhattan plots for all traits.
1.3 Files in the directory Data files produced from R scripts
ThefilesinthisfolderareproducedasaresultofexecutingtheRscriptsinthedirectoryASReml interaction R scripts.
1. Output readindata aus – data files produced from executing the script readindata aus.R.
2. Output readindata ind – data files produced from executing the script readindata ind.R.
3. Output readindata ausindica pca–datafilesproducedfromexecutingthescriptreadindata ausindica pca.R.
4. Output analysedata aus – data files produced from executing the script analysedata aus.R.
5. Output analysedata ind – data files produced from executing the script analysedata ind.R.
6. Output analysedata ausind pca – data files produced from executing the script analysedata ausind.R.
7. Output Maps aus, Output Maps indica and Output Maps INDAUS – data files produced from executing
the script map.R.
8. Output Manplots aus,Output Manplots indicaandOutput Manplots INDAUS–datafilesproducedfrom
executing the script Manplots.R.
References
Al-Tamimi, N., C.J.Brien, H.Oakey, B.Berger, S.Saade, Y.S.Ho, S.M.Schm¨ockel, M.Tester, andS.Negr˜ao
(2016) Salinity tolerance loci revealed in rice using high-throughput non-invasive phenotyping. Nature Com-
munications,. accepted for publication.
Butler, D., B. R. Cullis, A. R. Gilmour, and B. J. Gogel (2009) Analysis of Mixed Models for S language
environments: ASReml-R reference manual.. Brisbane: DPI Publications.
Chen, W. et al. (2011) Genome-wide association analyses provide genetic and biochemical insights into natural
variation in rice metabolism. Nature Genetics, 46, 714–721.
Dabney,A.andJohnD.Storey,J.D.andG.R.Warnes(2004) qvalue: Q-valueestimationforfalsediscoveryrate
control. URL: http://cran.at.r-project.org/src/contrib/Archive/qvalue/, (R package version 1.26.0,
accessed August 3, 2016).
McCouch, S. R. (2016) Open access resources for genome-wide association mapping in rice. Nature Communi-
cations, 7, 10532.
R Development Core Team (2015) R: A Language and Environment for Statistical Computing. Vienna, Austria:
R Foundation for Statistical Computing. URL: http://www.r-project.org.
Turner, S. (2014) qqman: Q-Q and manhattan plots for GWAS data. URL: http://cran.at.r-project.org/
web/packages/qqman/index.html, (R package version 0.1.2, accessed May 12, 2016).
2
--- page 4 ---
Yang, W. et al. (2014) Combining high-throughput phenotyping and genome-wide association studies to reveal
natural genetic variation in rice. Nature Communications, 5, 5087.
Yu, J. et al. (2006) A unified mixed-model method for association mapping that accounts for multiple levels of
relatedness. Nature Genetics, 38, 203–208.
Zhao, K. et al. (2014) Genome-wide association mapping reveals a rich genetic architecture of complex traits in
Oryza sativa. Nature Communications, 2, 467.
3