#'# Script with spin commands to process Stephanie Saade's data
#'## A good way to create the output file is to open the md file produced in spinning the file. 
#'## Then use Preview pdf 
options(java.parameters = "-Xmx10g", width=100)
library(knitr)
library(XLConnect)
library(imageData)

#'# Generate raw.dat
knitr::spin("Rice2015.raw.v8.r")
save.image("data.RData")

#'# Generate longi.dat
knitr::spin("Rice2015.longi.v8.r")
head(longi.0254.dat)
head(longi.0255.dat)
save.image("data.RData")

#'# Draw some plots of Area.smooth, AGR and RGR for different df
load("data.RData")
knitr::spin("Rice2015.dfcheck.v3.r")
save.image("data.RData")

#'# Produce longitudal plots
load("data.RData")
knitr::spin("Rice2015.plots.v8.r")
save.image("data.RData")

#'# Create cart.dat
knitr::spin("Rice2015.cart.v10.r")
#Check number of rows when the extra carts with Barke are excluded 
# - should be 2 x 24 x 22 = 1056
nrow(cart.0254.dat)
nrow(cart.0255.dat)
head(cart.0254.dat)
head(cart.0255.dat)
save.image("data.RData")

#'# Plot growth rate correlations
load("data.RData")
knitr::spin("Rice2015.cart.plots.v7.r")
save.image("data.RData")

#'# Create SIIT.dat
load("data.RData")
knitr::spin("Rice2015.SIIT.v6.r")
head(longi.0254.SIIT.dat)
head(cart.0254.SIIT.dat)
head(longi.0255.SIIT.dat)
head(cart.0255.SIIT.dat)
save.image("data.RData")

#'# Create Excel spreadsheets - need to delete files before running this
"createXLfile"<- function(excel.file, longi.dat, longi.SIIT.dat, cart.dat, cart.SIIT.dat)
{ wb = loadWorkbook(excel.file, create=TRUE)
  createSheet(wb, name=c("Longitudinal data","Longitudinal SIIT data","Cart data","Cart SIIT data"))
  writeWorksheet(wb,  longi.dat, sheet="Longitudinal data")
  writeWorksheet(wb,  longi.SIIT.dat, sheet="Longitudinal SIIT data")
  writeWorksheet(wb,  cart.dat, sheet="Cart data")
  writeWorksheet(wb,  cart.SIIT.dat, sheet="Cart SIIT data")
  nc <- ncol(longi.dat)
  n <- nrow(longi.dat)
  k <- match("Snapshot.Time.Stamp", names(longi.dat))
  setColumnWidth(wb, sheet = "Longitudinal data", column = 1:nc, width = 3500)
  setColumnWidth(wb, sheet = "Longitudinal data", col = k, width = 5000)
  csDate = createCellStyle(wb, name = "date")
  setDataFormat(csDate, format = "yyyy-mm-dd h:mm:ss")
  setCellStyle(wb, sheet = "Longitudinal data", row=2:n, col = k, cellstyle = csDate)
  nc <- ncol(cart.dat)
  setColumnWidth(wb, sheet = "Cart data", column = 1:nc, width = 3500)
  nc <- ncol(longi.SIIT.dat)
  setColumnWidth(wb, sheet = "Longitudinal SIIT data", column = 1:nc, width = 5000)
  nc <- ncol(cart.SIIT.dat)
  setColumnWidth(wb, sheet = "Cart SIIT data", column = 1:nc, width = 5000)
  saveWorkbook(wb)
  invisible()
}

load("data.RData")
excel.file <- file.path("Rice2015.0254.v8.xlsx")
createXLfile(excel.file, longi.0254.dat, longi.0254.SIIT.dat, cart.0254.dat, cart.0254.SIIT.dat)

excel.file <- file.path("Rice2015.0255.v8.xlsx")
createXLfile(excel.file, longi.0255.dat, longi.0255.SIIT.dat, cart.0255.dat, cart.0255.SIIT.dat)
