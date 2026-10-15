options(java.parameters = "-Xmx11g")
library(knitr)
library(imageData)
library(dae)
opts_chunk$set("tidy" = FALSE) 

#'# Step 7: Form continuous and interval SIITs
#'
#'## 7a) Calculate continuous SIITs
#+
cols.retained <-  c("Snapshot.ID.Tag","Smarthouse","Lane","Position",
                    "Days","Snapshot.Time.Stamp", "Hour", "xDays",
                    "Zones","xZones","SHZones","ZLane","ZMainplots",
                    "xMainPosn", "Genotype.ID")
responses.GR <- c("Area.smooth.AGR","Area.smooth.AGR","Area.smooth.RGR")
suffices.results <- c("diff", "SIIT", "SIIT")
responses.SIIT <- unlist(Map(paste, responses.GR, suffices.results,sep="."))

#'## continuuous SIITs for 0254
#+ 
longi.0254.SIIT.dat <- 
               twoLevelOpcreate(responses.GR, longi.0254.dat, suffices.treatment=c("C","S"),
                                operations = c("-", "/", "/"), suffices.results = suffices.results, 
                                columns.retained = cols.retained, 
                                by = c("Smarthouse","Zones","ZMainplots","Days"))
longi.0254.SIIT.dat <- with(longi.0254.SIIT.dat, 
                            longi.0254.SIIT.dat[order(Smarthouse,Zones,ZMainplots,Days),])
save(longi.0254.SIIT.dat, file="03-longi.0254.SIIT.dat.rda")

#'## continuuous SIITs for 0255
#+ 
longi.0255.SIIT.dat <- 
              twoLevelOpcreate(responses.GR, longi.0255.dat, suffices.treatment=c("C","S"),
                                operations = c("-", "/", "/"), suffices.results = suffices.results, 
                                columns.retained = cols.retained, 
                                by = c("Smarthouse","Zones","ZMainplots","Days"))
save(longi.0255.SIIT.dat, file="03-longi.0255.SIIT.dat.rda")

#' ### Plot SIIT profiles for each experiment 
#' 
#+ "03-SIITProfiles"
k <- 2
nresp <- length(responses.SIIT)
limits <- with(longi.0254.SIIT.dat, list(c(min(Area.smooth.AGR.diff, na.rm=TRUE),max(Area.smooth.AGR.diff, na.rm=TRUE)),
                                         c(0,3),
                                         c(0,1.5)))
#Plots for 0254
for (k in 1:nresp)
{ plt <- longiPlot(data = longi.0254.SIIT.dat, x="xDays+35.42857143", response = responses.SIIT[k], 
                   y.title=responses.SIIT[k], facet.x="Smarthouse", facet.y=".", printPlot=FALSE, )
  plt <- plt + geom_vline(xintercept=29, linetype="longdash", size=1) +
               scale_x_continuous(breaks=seq(28, 42, by=2)) + 
               scale_y_continuous(limits=limits[[k]])
  print(plt)
}

#Plots for 0255
for (k in 1:nresp)
{ plt <- longiPlot(data = longi.0255.SIIT.dat, x="xDays+35.42857143", response = responses.SIIT[k], 
                   y.title=responses.SIIT[k], facet.x="Smarthouse", facet.y=".", printPlot=FALSE)
  plt <- plt + geom_vline(xintercept=29, linetype="longdash", size=1) +
               scale_x_continuous(breaks=seq(28, 42, by=2)) + 
               scale_y_continuous(limits=limits[[k]])
  print(plt)
}

#'## 7b) Calculate interval SIITs and check for large values for SIIT for Days 31to35
suffices <- list("31to35","35to38","31to38","38to42")
response <- "Area.smooth.RGR.31to35"
SIIT.anom <- function(response, data)
{ SIIT <- paste(response, "SIIT", sep=".")
  tmp<-na.omit(data)
  print(summary(tmp[SIIT]))
  big.SIIT <- with(tmp, tmp[tmp[SIIT] > 1.15, c("Snapshot.ID.Tag.C","Genotype.ID",
                                                paste(response,"C",sep="."), 
                                                paste(response,"S",sep="."), SIIT)])
  big.SIIT <- big.SIIT[order(big.SIIT[SIIT]),]
  print(big.SIIT)
  plt <- ggplot(tmp, aes_string(SIIT)) +
    geom_histogram(aes(y = ..density..), binwidth=0.05) +
    geom_vline(xintercept=1.15, linetype="longdash", size=1) +
    theme_bw() + facet_grid(Smarthouse ~.)
  print(plt)
  plt <- ggplot(tmp, aes_string(x="Smarthouse", y=SIIT)) +
    geom_boxplot() + theme_bw()
  print(plt)
  invisible(big.SIIT)
}
responses.SIITinterval <- as.vector(outer("Area.smooth.RGR", suffices, paste, sep="."))

#'## SIITs for 0254
#+ "01-0254-SIITIntClean"
cart.0254.SIIT.dat <- twoLevelOpcreate(responses.SIITinterval, cart.0254.dat, 
                                       suffices.treatment=c("C","S"), 
                                       suffices.results="SIIT", 
                                       columns.suffixed="Snapshot.ID.Tag")
save(cart.0254.SIIT.dat, file="03-cart.0254.SIIT.dat.rda")
SIIT.anom(response, cart.0254.SIIT.dat)

#'## SIITs for 0255
#+ "01-0255-SIITIntClean"
cart.0255.SIIT.dat <- twoLevelOpcreate(responses.SIITinterval, cart.0255.dat, 
                                       suffices.treatment=c("C","S"), 
                                       suffices.results="SIIT", 
                                       columns.suffixed="Snapshot.ID.Tag")
save(cart.0255.SIIT.dat, file="03-cart.0255.SIIT.dat.rda")
SIIT.anom(response, cart.0255.SIIT.dat)
