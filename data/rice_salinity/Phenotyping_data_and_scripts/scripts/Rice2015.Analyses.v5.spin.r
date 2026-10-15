#'# Script with spin commands to do mixed model analyses with Lines fixed for the Rice 2015 Salt Experiment data
#'## A good way to create the output file is to open the md file produced in spinning the file. 
#'## Then use Preview pdf from within R-Studio
options(java.parameters = "-Xmx10g", width=100)
library(knitr)
library(XLConnect)
library(asreml)
library(asremlPlus)
opts_chunk$set("tidy" = FALSE, "dev" = "png", "dpi" = 600, 
               "fig.width" = 9, "fig.height" = 9)

#'# Analyse the AGRs and RGRs for specified intervals
load(file="02-cart.0254.dat.rda")
load(file="02-cart.0255.dat.rda")
knitr::spin("Rice2015.cart.grates.fix.v5.r")

#'# Analyse the SIITs for specified intervals calculated from the AGRs and RGRs 
load(file="03-cart.0254.SIIT.dat.rda")
load(file="03-cart.0255.SIIT.dat.rda")
knitr::spin("Rice2015.SIIT.fix.v5.r")

#'# Save image
save.image("data.Analyses.RData")
load("data.Analyses.RData")
