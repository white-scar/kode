#'# Script to import Excel files of raw data and do some editing before saving
#'### This version makes sure that there is a row for every combinations of 24 Lanes x 23 Positions 
options(java.parameters = "-Xmx10g", warn=0)
library(knitr)
library(imageData)
library(dae)
library(XLConnect)
opts_chunk$set("tidy" = FALSE) 

#'# Step 1: Import imaging file to produces raw.dat
#+ "ImageTimes"
#Data set 0254 _ Indica panel
raw.0254.dat <- suppressWarnings(importExcel(file = "0254 rawdata.xlsx", sheet="0254 rawdata", 
                                             planting.time = "2015-02-11 0:00 AM"))

#Set observations for day 40 to missing and Time.after.Planting..d. to 40 
day40Delay <- (with(raw.0254.dat, Snapshot.Time.Stamp > as.POSIXct("2015-03-23 0:00 AM") & 
                                 (Snapshot.Time.Stamp < as.POSIXct("2015-03-24 03:30 AM"))))
ncinput <- ncol(raw.0254.dat)
raw.0254.dat[day40Delay, match("Projected.Shoot.Area..pixels.", names(raw.0254.dat)):ncinput] <- NA
raw.0254.dat[day40Delay, "Time.after.Planting..d."] <- 40
with(raw.0254.dat, table(Time.after.Planting..d.))


## Set up list of plants to be deleted and then set their data to missing
ncinput <- ncol(raw.0254.dat)
#excl.0254 <- c("045234-C", "044441-C", "044620-C", "045004-S")
excl.0254 <- c("044441-C", "044620-C", "044361-S", "044689-S", "045004-S")
raw.0254.dat[raw.0254.dat$Snapshot.ID.Tag %in% excl.0254, 
             match("Weight.Before", names(raw.0254.dat)):ncinput] <- NA
## Set Gentoypes with one or more plants with a large OST and set their data to missing
excl.0254.OST <- c("42019")
raw.0254.dat[raw.0254.dat$Genotype.ID %in% excl.0254.OST, 
             match("Weight.Before", names(raw.0254.dat)):ncinput] <- NA
#Data deleted for:
table(raw.0254.dat[raw.0254.dat$Snapshot.ID.Tag %in% excl.0254,]$Snapshot.ID.Tag )  
table(raw.0254.dat[raw.0254.dat$Genotype.ID %in% excl.0254.OST,]$Genotype.ID ) 
with(raw.0254.dat, table(Time.after.Planting..d.))
##Delete Weight.Before and Weight.After values equal to -999
raw.0254.dat[!is.na(raw.0254.dat$Weight.Before) & raw.0254.dat$Weight.Before < 0, 
             c("Weight.Before", "Weight.After")] <- NA

#Data set 0255 - AUD panel
raw.0255.dat <- suppressWarnings(importExcel(file = "0255 rawdata.xlsx", sheet="0255 rawdata"))
## Get list of plants to be deleted and set their data to missing
ncinput <- ncol(raw.0255.dat)
#excl.0255 <- c("045980-C", "045928-S", "045593-S", "045638-S", "046012-C", "046013-S", "046495-S")
excl.0255 <- c("045980-C", "045928-S", "046012-C", "046029-C", "045567-C")
raw.0255.dat[raw.0255.dat$Snapshot.ID.Tag %in% excl.0255, 
             match("Weight.Before", names(raw.0255.dat)):ncinput] <- NA
## Set Gentoypes with one or more plants with a large OST and set their data to missing
excl.0255.OST <- c("121050")
raw.0255.dat[raw.0255.dat$Genotype.ID %in% excl.0255.OST, 
             match("Weight.Before", names(raw.0255.dat)):ncinput] <- NA
#Data deleted for:
table(raw.0255.dat[raw.0255.dat$Snapshot.ID.Tag %in% excl.0255,]$Snapshot.ID.Tag ) 
table(raw.0255.dat[raw.0255.dat$Genotype.ID %in% excl.0255.OST,]$Genotype.ID ) 
with(raw.0255.dat, table(Time.after.Planting..d.))
##Delete Weight.Before and Weight.After values equal to -999
raw.0255.dat[!is.na(raw.0255.dat$Weight.Before) & raw.0255.dat$Weight.Before < 0, 
             c("Weight.Before", "Weight.After")] <- NA

#'## Save raw data
save(raw.0254.dat, file="00-raw.0254.dat.rda")
save(raw.0255.dat, file="00-raw.0255.dat.rda")

