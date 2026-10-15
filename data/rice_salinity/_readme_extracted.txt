================================================================================
D:\pythonproject\knode\data\rice_salinity\README_for_Phenotyping_data_and_scripts.pdf
--- page 1 ---
Data and script files for the preparation and pheotypic analysis for the main
traits and their derived indices for the 2015 salt experiments on the indica and
aus Rice panels reported in Al-Tamimi et al. (2016)
C. J. Brien1, N. Al-Tamimi, H. Oakey B. Berger, S. Saade, Y. S. Ho, S. M. Schmo¨ckel, M.
Tester2 and S. Negra˜o
1email: chris.brien@unisa.edu.au
2email: Mark.Tester@KAUST.EDU.SA
August 5, 2016
--- page 2 ---
This document describes the data and code files that were used in the data preparation and phenotypic
analysis phases for the two experiments reported by Al-Tamimi et al. (2016): experiment 0254 that investi-
gates an indica panel and experiment 0255 that investigates an aus panel. These files have been provided in
Phenotyping data and scripts.zip.
The main objective of the data preparation phase is to produce growth rates, associated Shoot Ion Indepen-
dent Tolerance (SIIT) indices and transpiration rates and indices for each experiment. The growth rates are
calculatedfromtherawdataandfromdatasmoothedusingnaturalcubicsplines. Theyarecalculatedusingthe
differences between successive imagings divided by the number of days between the imagings. For the Absolute
Growth Rates (AGR); the difference between the values of the response at the two times is taken; while for the
Relative Growth Rates (RGR), the difference between the logarithms of the values of the response at the two
times is taken. Transpiration is calculated by taking water loss for a pot with a plant and adjusting it for the
water loss in the nearest evaporation pot (pot without a plant).
In the phenotypic analysis phase, the traits are subject to a mixed model analyses to produce spatially
adjusted BLUEs for each line-treatment combination for each trait. The resulting BLUEs are then subject to
association analyses, one for each panel and another for the combined single set. The data and scripts for the
latter are available in GWAS data and scripts.zip.
1 Files in Phenotyping data and scripts.zip
1.1 Files in the root directory
This file: Phenotyping readme.pdf
1.2 Files in the directory data
Data input files: Files in csv and xlsx formats containing the original data.
(cid:136) 0254 rawdata.csv and 0255 rawdata.csv — csv files containing the raw data obtained from the Plant
Accelerator for the observed time points for the carts with plants.
(cid:136) 0254rawdata.xlsxand0255rawdata.xlsx—xlsxfilescontainingtherawdataobtainedfromthePlant
Accelerator for the observed time points for the carts with plants.
(cid:136) 0254evapdata.csvand0255evapdata.csv—csvfilescontainingtherawdataobtainedfromthePlant
Accelerator for the observed time points for the evaporation carts (carts with soil but no plants).
(cid:136) 0254 evapdata.xlsx and 0255 evapdata.xlsx — xlsx files containing the raw data obtained from the
PlantAcceleratorfortheobservedtimepointsfortheevaporationcarts(cartswithsoilbutnoplants).
(cid:136) Evap.carts.csv,Evap.carts.xlsxandEvap.carts.rda—acsv,anxlsxandanRdatafilewithinformation
about which evaporation cart is to be used to adjust water loss for each cart with a plant.
Data files containing the data produced: ThesefilesareproducedasaresultofexecutingtheRdataprepa-
ration scripts in Section 1.3.
1. 00-raw.0254.dat.rda and 00-raw.0255.dat.rda — R data files produced from executing the script
Rice2015.raw.v8.r.
2. 01-longi.0254.dat.rda and 01-longi.0255.dat.rda — R data files produced from executing the script
Rice2015.longi.v8.r.
3. 02-cart.0254.dat.rda and 02-cart.0255.dat.rda — R data files produced from executing the script
Rice2015.cart.v10.r.
4. 03-longi.0254.SIIT.dat.rda and 03-longi.0255.SIIT.dat.rda — R data files produced from executing
the script Rice2015.SIIT.v6.r.
5. Rice2015.0254.v8.xlsx, Rice2015.0255.v8.xlsx — Excel files created using the createXLfile function at
the end of Rice2015.spin.v10.r.
6. 0254 Cart BLUEs.v8.csv and from Rice2015.0254.v8.xlsx; 0255 Cart BLUEs.v8.csv and 0254 SIIT
BLUEs.v8.csv from Rice2015.0255.v8.xlsx.
1
--- page 3 ---
1.3 Files in the directory scripts
ThescriptscontaincodetobeexecutedusingtheRstatisticalcomputingenvironment(RDevelopmentCoreTeam,
2015).
R scripts for data preparation: The scripts can be run by executing the sequence of functions in
Rice2015.spin.v10.r. They use functions in the R package imageData.
1. Rice2015.spin.v10.r — the R script containing the sequence of calls to spin from the knitr package
(see further details below) to execute the other R scripts in this list.
2. Rice2015.raw.v8.r – an R script to import the Excel file containing the raw data from the Plant
Accelerator.
3. Rice2015.longi.v8.r — an R script to produce longitudinal values (values for each imaging day) for
traits measured in the Plant Accelerator.
4. Rice2015.dfcheck.v3.r — an R script to investigate various values of the smoothing parameter df to
be used in smoothing PSA.
5. Rice2015.plots.v8.r — an R script to produce plots of the longitudinal data and to identify possibly
anomalous plants.
6. Rice2015.cart.v10.r — an R script to produce a single value of each traits for each cart in the Plant
Accelerator.
7. Rice2015.cart.plots.v7.r — an R script to produce plots involving the traits for which there is a single
value only for each cart in the Plant Accelerator.
8. Rice2015.SIIT.v6.r—anRscripttoproducedifferencesandratiosofthevaluesofthePSAAGRand
RGR for each pair of carts in the Plant Accelerator. The ratios of the PSA RGR are the SIIT values.
R scripts for producing the spatially-adjusted BLUEs: Thescriptscanberunbyexecutingthesequence
of functions in Rice2015.Analyses.v5.spin.r. They use functions in the R packages asreml, asremlPlus,
dae, ggplot2, QTLRel and XLConnect.
1. Rice2015.Analyses.v5.spin.r — the R script containing the sequence of calls to spin from the knitr
package (see further details below) to execute the other R scripts in this list.
2. Rice2015.cart.grates.fix.v5.r — an R script to carry out the mixed linear model analysis of the AGRs
and RGRs based on Lines fixed and so to produce the BLUEs for them.
3. Rice2015.SIIT.fix.v5.r — an R script to carry out the mixed linear model analysis of the SIITs based
on Lines fixed and so to produce the BLUEs for them.
2 Data preparation phase
2.1 The process
ThedatawereinputintoRstatisticalcomputingenvironment(RDevelopmentCoreTeam,2015)andprocessed
with the aid of a set of functions collected into the R package imageData (Brien, 2016c), available at http://
cran.at.r-project.org/web/packages/imageData/index.html. Theprocessingwascarriedoutby‘spinning’
a sequence of R scripts (see Rice2015.spin.v10.r). To ‘spin’ a script is to execute the function spin from the
knitr package (Xie, 2015) with the name of the script supplied to the function. A ‘spin’ creates a html and
an md file, each of which contains the output from running the script. Other packages used include XLConnect
(MiraiSolutionsGmbHetal.,2015),ggplot2(Wickham,2016),GGally(Schloerkeetal.,2016),Hmisc(HarrelJr,
2016) and dae(Brien, 2016b), all of which can be downloaded from CRAN (http://www.r-project.org).
Five sets of data for each experiment, which are available in R data files, were produced in a seven-step
process. The five sets are: (i) Raw data, being the edited data obtained from the Plant Accelerator (step 1);
(ii) Longitudinal data, being measurements made on each cart over time (steps 2–4); (iii) Cart data, being a
single measurement on the plant in a cart (step 6); (iv) Longitudinal SIIT data, being differences or ratios of
2
--- page 4 ---
measurementsovertimecalculatedfromthetwoplantsinneighbouringcartsthatformamainplot(step7); (v)
Cart SIIT data, being the differences or ratios of measurements calculated from a single measurement on each
of the two plants in neighbouring carts that form a main plot (step 7).
The seven-step process, used to produce the data sets and which was repeated in several cycles, is:
1. The raw data is imported from the Excel file supplied by the Plant Accelerator staff and data editing
performed. Initially minor editing is performed. In particular, Time.after.Planting..d. is recalculated and
the PSA data for the imaging that was to be done on day 40 for Experiment 254 is set to missing because
of an equipment malfunction. Also, the name ‘final.edit.water.loss’ was changed to Water.Loss. Plants
that have died or otherwise indentified as being faulty are removed. At subsequent cycles further plants
may be excluded, using criteria described below. The script for doing this is Rice2015.raw.v8.r.
2. The data was reduced to a subset of the imaging variables produced by the Lemna Tec Scanalyzer and
covariates and factors added that might be used in the analysis. In 0254, because of a delay in imaging on
day 40, the traits for day 40 are removed, but with adjustment of the water traits. For these experiments,
the water calculations are made as follows: (i) the Evaporation data from the empty pots is imported;
(ii) the Water.Loss, Water.Amount and empty-pot Evaporation had 100 ml added to their day 30 values
because this amount of water was added but not recorded on the salting day (day 29); (iii) in experiment
254, the values of Water.Loss, Water.Amount and Evaporation for day 40 are added to those for day 41
and the values for day 40 set to missing (NA).
3. Derived traits, including continuous growth rates i.e. a growth rate for a cart for each time of observation,
exceptthefirst,andcumulativeresponsesareobtained. AlsoTranspirationiscalculatedforacartforeach
time of observation by subtracting from the Evaporation for the nearest evaporation cart from Water.Loss
for the cart . Then cumulative Transpiration and cumulative Transpiration Use Efficiency is calculated as
the Area at an imaging divided by the cumulative Transpiration. Finally, a continuous Transpiration Use
Index (TUE) is calculated as the Total growth since the previous imaging divided by the Transpiration
since the previous imaging.
4. Splines are fitted to smooth the longitudinal trends in Total Area and the Transpiration rate. The plots of
Area.smooth, AGR and RGR, when different df are used in obtaining Area.smooth, are examined. It was
decided to use df equal to 4. Continuous growth rates and TUE indices are calculated from the smoothed
Total Area and Transpiration rate. Steps 2–4 are achieved using Rice2015.longi.v8.r.
5. An exploratory examination of the unsmoothed data is performed by producing longitudinal plots of
unsmoothed and smoothed responses (see Rice2015.plots.v8.r). An investigation of anomalous plants is
alsocarriedout. Therulesusedtoidentifyanomalousplantstobefurtherinvestigatedforpossibleexclusion
are:
(cid:136) the plant has an Area.smooth value in the interval from Day 40 onwards that is less than 100 for
Control plants and less than 75 for Salted plants;
(cid:136) the plant has an Area.smooth AGR value in the interval from Day 40 onwards that is less than 2.5;
(cid:136) the plant has an Area.smooth.RGR value in the interval from Day 40 onwards that is less than 0.08
for Control plants and less than 0.01 for Salted plants.
6. Produce cart data: traits for which there is a single value for each Snapshot.ID.Tag or cart are obtained.
Thisincludesdataforasingleimaging(e.g. thelastdateofimaging)andgrowthratesandTUEsforspeci-
fiedintervals. Responseswereproducedforthefollowingintervals: Days31–35,Days35–38,Days31–38and
Days38–42. ThisisdonewiththescriptRice2015.cart.v10.r. AlsoproducedusingRice2015.cart.plots.v7.r
are scatterplots, with correlation coeffcients, illustrating the relationships between values for different in-
tervals of AGRs, RGRs and TUEs.
7. Produce SIIT data: for each main plot, obtain continuous differences between the treatments for the AGR
from the smoothed Total area and continuous SIIT indices for the the AGR and RGR from the smoothed
Total area. The script for this is Rice2015.SIIT.v5.r.
3
--- page 5 ---
If, in carrying out the process, plants are identified for exclusion, the above seven-step process is repeated
with the the newly-identified plants excluded at step 1. The included scripts are those for the final cycle of the
process.
In the end, plants were excluded for the following reasons:
1. Plants that were noted to be dying or dead, or were not loaded;
2. Plants suspected of receiving extra water because of roof leakage;
3. Plants that had not received the correct salt treatment;
4. ControlplantsthatmarkedlyoutperformedbyitsassociatedSalt-treatedplant(SIITfordays31–35greater
than 1.25)
2.2 Plots produced
Traits have been plotted over time for each plant in longitudinal or profile plots, each of which contains four
graphs for a trait: one for each combination of Smarthouse and treatment. In each graph, a line represents the
values for a single plant. These plots are produced using Rice2015.plots.v8.r. For each experiment, the plots are
organised as follows:
Raw or unsmoothed data Plots for the following responses are given:
1. PSA (kpixels)
2. PSA AGR (kpixels per day)
3. PSA RGR (per day)
4. Transpiration rate (mL)
5. Total area TUE (kpixels per mL) = PSA / Transpiration rate
6. Cumulative transpiration (mL)
7. Cumulative Transpiration Use Efficiency = PSA / cumulative Transpiration (kpixels per mL)
Smoothed data Plots are given for the PSA (kpixels) and their AGRS and RGRs and for the Transpiration
rate and Total area TUE.
Investigation of anomalies for smoothed Total Area Up to seven plots, as well as a listing of anomalous
plants,areprovided. Anomalousplantsarethosewith(i)asmoothedTotalareafromDay40onwardsless
than 100 for Control plants and less than 75 for Salted plants, (ii) an AGR from Day 40 onwards less than
2.5 and (iii) an RGR from Day 40 onwards less than 0.08 for Control plants and less than 0.01 for Salted
plants. Plots for anomalous plants are omitted if there are no such plants.
1. Plot of Area.smooth (smoothed PSA) with anomalous plants using (i) removed
2. Plot of Area.smooth (smoothed PSA) for only anomalous plants according to (i)
3. Plot of the AGR with anomalous plants using (ii) removed
4. Plot of the AGR for only anomalous plants according to (ii)
5. Plot of the RGR with anomalous plants using (iii) removed
6. Plot of the RGR for only anomalous plants according to (iii)
7. Plot of Total Area (PSA) values with anomalous plants highlighted in purple
4
--- page 6 ---
3 Phenotypic analysis phase
3.1 Producing the spatially-adjusted BLUEs
The data contained in 02-cart.0254.dat.rda and 02-cart.0255.dat.rda are analysed using the script
Rice2015.cart.grates.fix.v5.r to produce the BLUEs stored in sheets in Rice2015.0254.v8.xlsx and
Rice2015.0255.v8.xlsx. The data contained in 03-longi.0254.SIIT.dat.rda and 03-longi.0255.SIIT.dat.rda are
analysed using the script Rice2015.SIIT.fix.v5.r to produce the BLUEs stored in sheets in Rice2015.0254.v8.xlsx
and Rice2015.0255.v8.xlsx. In doing this the following R packages (R Development Core Team, 2015) are used:
ASReml-R (Butleretal., 2009), asremlPlus(Brien, 2016a), dae (Brien,2016b), ggplot2 (Wickham,2016), QTL-
Rel (Cheng, 2015) and XLConnect (Mirai Solutions GmbH et al., 2015).
References
Al-Tamimi, N., C.J.Brien, H.Oakey, B.Berger, S.Saade, Y.S.Ho, S.M.Schm¨ockel, M.Tester, andS.Negr˜ao
(2016) Salinity tolerance loci revealed in rice using high-throughput non-invasive phenotyping. Nature Com-
munications,. accepted for publication.
Brien, C. J. (2016a) asremlPlus: augments the use of asreml in fitting mixed models. URL: http://cran.at.
r-project.org/web/packages/asremlPlus/index.html, (R package version 2.0-9, accessed May 17, 2016).
Brien, C. J. (2016b) dae: functions useful in the design and ANOVA of experiments. URL: http://cran.at.
r-project.org/web/packages/dae/index.html, (R package version 2.7-16, accessed May 17, 2016).
Brien,C.J.(2016c)imageData: aidsinprocessingandplottingdatafromaLemna-TecScananalyzer.URL:http:
//cran.at.r-project.org/web/packages/imageData/index.html,(Rpackageversion0.1-21,accessedMay
17, 2016).
Butler, D., B. R. Cullis, A. R. Gilmour, and B. J. Gogel (2009) Analysis of Mixed Models for S language
environments: ASReml-R reference manual.. Brisbane: DPI Publications.
Cheng,R.(2015)QTLRel: ToolsforMappingofQuantitativeTraitsofGeneticallyRelatedIndividualsandCalcu-
lating Identity Coefficients from Pedigrees. URL: http://cran.at.r-project.org/web/packages/QTLRel/
index.html, (R package version 0.2-15, accessed May 12, 2016).
Harrel Jr, F. E. (2016) Hmisc: Harrel miscellaneous. URL: http://cran.at.r-project.org/web/packages/
Hmisc/index.html, (R package version 3.17-4, accessed August 3, 2016).
Mirai Solutions GmbH, M. Studer, The Apache Software Foundation, and S. Colebourne (2015) XLConnect:
Excel connector for R. URL: http://cran.at.r-project.org/web/packages/XLConnect/index.html, (R
package version 0.2-11, accessed May 12, 2016).
R Development Core Team (2015) R: A Language and Environment for Statistical Computing. Vienna, Austria:
R Foundation for Statistical Computing. URL: http://www.r-project.org.
Schloerke, B. et al. (2016) GGally: Extension to ‘ggplot2’. URL: http://cran.at.r-project.org/web/
packages/GGaly/index.html, (R package version 1.2.01, accessed August 3, 2016).
Wickham, H. (2016) ggplot2: An Implementation of the Grammar of Graphics. URL: http://cran.at.
r-project.org/web/packages/ggplot2/index.html, (R package version 2.1.0, accessed May 12, 2016).
Xie, Y.(2015) knitr: A General-Purpose Package for Dynamic Report Generation in R. Rpackageversion1.11.
URL: http://yihui.name/knitr/.
5
================================================================================
D:\pythonproject\knode\data\rice_salinity\README_for_GWAS_data_and_scripts.pdf
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