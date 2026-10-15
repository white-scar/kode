#'# Step 5: Do exploratory plots on unsmoothed and smoothed longitudinal data

#'### Set up
#+
library(imageData)
library(dae)
library(knitr)
library(ggplot2)
opts_chunk$set("tidy" = FALSE, "dev" = "png", "dpi" = 300, 
               "fig.width" = 9, "fig.height" = 9)#,
#               "out.width" = 1200, "out.height" = 1200) 

responses.longi <- c("Area","Area.AGR","Area.RGR",
                     "Transpiration.rate", "Area.TUE", "Transpiration.Cum", "TUE.cum")
responses.smooth <- c("Area.smooth","Area.smooth.AGR","Area.smooth.RGR",
                      "Transpiration.rate.smooth", "Area.smooth.TUE")
titles <- c("PSA (kpixels)", "PSA AGR (kpixels per day)", "PSA RGR (per day)", 
            "TR (mL water transpired per plant per day)", 
            "TUE (increase in kpixels per mL water transpired)", 
            "Cumulative transpiration (mL)", 
            "Cumulative TUE (kpixels per mL)")
cols.output <- c("Snapshot.ID.Tag", "Smarthouse", "Lane", "Position", 
                 "Treatment.1", "Genotype.ID", "Days")

titles.smooth<-titles[c(1:5)]
"profilePlots" <- function(longi.dat, responses.longi, responses.smooth, titles, titles.smooth, ...)
{ #' ### Plot unsmoothed profiles for all longitudinal  responses 
  #' 
  #+ "01-ProfilesAll"
  cat("\n\n#### Profiles for unsmoothed data ####\n\n")
  k <- 1
  nresp <- length(responses.longi)
  limits <- list(c(0,1000), c(-50,125), c(-0.05,0.40), c(0,400), c(-1,2.5), c(0,3000), c(0,4))
  klimit <- 0
  for (k in 1:nresp)
  { klimit <- klimit + 1
    plt <- longiPlot(data = longi.dat, response = responses.longi[k], 
                     y.title = titles[k], printPlot=FALSE, ...)
    plt <- plt + geom_vline(xintercept=29, linetype="longdash", size=1) +
                 scale_x_continuous(breaks=seq(28, 42, by=2)) + 
                 scale_y_continuous(limits=limits[[klimit]])
    print(plt)
  }
  
  #' ### Plot smoothed profiles for all longitudinal responses- GRs by differences
  #' 
  #+ "01-SmoothedProfilesAll"
  cat("\n\n#### Profiles for smoothed data ####\n\n")
  nresp.smooth <- length(responses.smooth)
  limits <- list(c(0,1000), c(0,100), c(0.0,0.40), c(0,300), c(0,1))
  for (k in 1:nresp.smooth)
  { plt <- longiPlot(data = longi.dat, response = responses.smooth[k], 
                     y.title = titles.smooth[k], printPlot=FALSE, ...)
    plt <- plt + geom_vline(xintercept=29, linetype="longdash", size=1) +
                 scale_x_continuous(breaks=seq(28, 42, by=2)) + 
                 scale_y_continuous(limits=limits[[k]])
    print(plt)
  }
  invisible()
}


#'# Plot profiles for 0254
#+ "01-0254-Profiles"
profilePlots(longi.0254.dat, responses.longi, responses.smooth, titles, titles.smooth, 
             x="xDays+35.42857143")

#'## Investigation of anomalies for 0254
#'### Area anomalies - plot without anomalous plants followed by plot of anomalous plants
#+ "01-0254-Areaanomalies"
response <- "Area.smooth"
anomalous <- anomPlot(longi.0254.dat, response=response, x = "xDays+35.42857143", vertical.line=29, 
                      groupsFactor="Treatment.1", lower=c(100, 75), start.time=40, 
                      breaks=seq(28, 42, by=2), whichPrint=c("innerPlot"),
                      y.title=response)
runt <- intervalValueCalculate(response=response, FUN = "anom", data=longi.0254.dat,
                               lower=13, end.time=28)
anomalous$data$Area.smooth.anom <- anomalous$data$Area.smooth.anom | runt$Area.smooth.anom
subs <- subset(anomalous$data, Area.smooth.anom & (Days==28 | Days==42))
if (nrow(subs) == 0)
{  cat("\n#### No anomalous data here\n\n")
} else
{ subs <- subs[order(subs["Smarthouse"],subs["Treatment.1"], subs[response]),]
  rownames(subs) <- NULL
  print(subs[c(cols.output, response)])
  anom.ID <- unique(subs$Snapshot.ID.Tag)
  outerPlot <- anomalous$outerPlot  + geom_text(data=subset(subs, Days==42),
                                                aes_string(x = "xDays+35.42857143", y = response, 
                                                           label="Snapshot.ID.Tag"), 
                                                size=3, hjust=0.7, vjust=0.5)
  print(outerPlot)
}

#'### AGR anomalies - plot without anomalous plants followed by plot of anomalous plants
#+ "01-0254-AGRanomalies"
anom.ID <- vector(mode = "character", length = 0L)
response <- "Area.smooth.AGR"
anomalous <- anomPlot(longi.0254.dat, response=response, lower=2.5, start.time=40, 
                      x = "xDays+35.42857143", vertical.line=29, breaks=seq(28, 42, by=2), 
                      whichPrint=c("innerPlot"), y.title=response)
subs <- subset(anomalous$data, Area.smooth.AGR.anom & Days==42)
if (nrow(subs) == 0)
{  cat("\n#### No anomalous data here\n\n")
} else
{ subs <- subs[order(subs["Smarthouse"],subs["Treatment.1"], subs[response]),]
  print(subs[c(cols.output, response)])
  anom.ID <- unique(c(anom.ID, subs$Snapshot.ID.Tag))
  outerPlot <- anomalous$outerPlot  + geom_text(data=subs,
                                                aes_string(x = "xDays+35.42857143", 
                                                           y = response, 
                                                           label="Snapshot.ID.Tag"), 
                                                size=3, hjust=0.7, vjust=0.5)
  print(outerPlot)
}
#'### RGR anomalies - plot without anomalous plants followed by plot of anomalous plants
#+ "01-0254-RGRanomalies"
response <- "Area.smooth.RGR"
anomalous <- anomPlot(longi.0254.dat, response=response, x = "xDays+35.42857143", 
                      groupsFactor="Treatment.1", lower=c(0.08,0.01), start.time=40, 
                      breaks=seq(28, 42, by=2), vertical.line=29, whichPrint=c("innerPlot"),
                      y.title=response)
subs <- subset(anomalous$data, 
               Area.smooth.RGR.anom & 
                 ((Smarthouse=="NW" & Treatment.1=="C" & Days==14) |
                    (Smarthouse=="NW" & Treatment.1=="S" & (Days==13 | Days==22)) | 
                    (Smarthouse=="NE" & Treatment.1=="C" & Days==22) |
                    (Smarthouse=="NE" & Treatment.1=="S" & Days==35)))
if (nrow(subs) == 0)
{  cat("\n#### No anomalous data here\n\n")
} else
{ subs <- subs[order(subs["Smarthouse"],subs["Treatment.1"], subs[response]),]
  print(subs[c(cols.output, response)])
  anom.ID <- unique(c(anom.ID, subs$Snapshot.ID.Tag))
  outerPlot <- anomalous$outerPlot + geom_text(data=subs,
                                               aes_string(x = "xDays+35.42857143", 
                                                          y = response, 
                                                          label="Snapshot.ID.Tag"), 
                                               size=3, hjust=0.5, vjust=0.5) 
  print(outerPlot)
}

#'### Plot of smoothed Total Area with tags that seem anomalous highlighted
#+ "01-0254-AreaReplot"
longi.0254.dat$anom <- longi.0254.dat$"Snapshot.ID.Tag" %in% anom.ID
plt <- longiPlot(data = longi.0254.dat, x="xDays+35.42857143", response = "Area.smooth", 
                 y.title = "Total area (1000 pixels)", 
                 colour.column="anom", colour.values=c("grey90", "purple"), 
                 alpha=0.5, printPlot=FALSE)
plt <- plt + geom_vline(xintercept=29, linetype="longdash", size=1) +
             scale_x_continuous(breaks=seq(28, 42, by=2))
if (length(anom.ID) > 0)
{ plt <- plt +  geom_text(data=subset(longi.0254.dat, anom & Days == 36),
                          aes_string(x = "xDays+35.42857143", y = "Area.smooth", 
                                     label="Snapshot.ID.Tag"), 
                          size=3, hjust=0.7, vjust=0.5) 
}
print(plt)
if (any(longi.0254.dat$anom))
{  print(subset(longi.0254.dat, anom & Days == 36)[c(cols.output,"Area.smooth")])
} else
  cat("\n#### No anomalous data here\n\n")


#'# Plot profiles for 0255
#+ "01-0255-Profiles"
profilePlots(longi.0255.dat, responses.longi, responses.smooth, titles, titles.smooth, 
             x="xDays+35.42857143")

#'## Investigation of anomalies for 0255
#'### Area anomalies - plot without anomalous plants followed by plot of anomalous plants
#+ "01-0255-Areaanomalies"
anom.ID <- vector(mode = "character", length = 0L)
response <- "Area.smooth"
anomalous <- anomPlot(longi.0255.dat, response=response, x = "xDays+35.42857143", 
                      groupsFactor="Treatment.1", lower=c(100, 75), start.time=40, 
                      vertical.line=29, breaks=seq(28, 42, by=2), whichPrint=c("innerPlot"),
                      y.title=response)
runt <- intervalValueCalculate(response=response, FUN = "anom", data=longi.0255.dat,
                               lower=13, end.time=28)
anomalous$data$Area.smooth.anom <- anomalous$data$Area.smooth.anom | runt$Area.smooth.anom
subs <- subset(anomalous$data, Area.smooth.anom & (Days==28 | Days==42))
if (nrow(subs) == 0)
{  cat("\n#### No anomalous data here\n\n")
} else
{ subs <- subs[order(subs["Smarthouse"],subs["Treatment.1"], subs[response]),]
  rownames(subs) <- NULL
  print(subs[c(cols.output, response)])
  anom.ID <- unique(subs$Snapshot.ID.Tag)
  outerPlot <- anomalous$outerPlot  + geom_text(data=subset(subs, Days==42),
                                                aes_string(x = "xDays+35.42857143", y = response, 
                                                           label="Snapshot.ID.Tag"), 
                                                size=3, hjust=0.7, vjust=0.5)
  print(outerPlot)
}

#'### AGR anomalies - plot without anomalous plants followed by plot of anomalous plants
#+ "01-0255-AGRanomalies"
response <- "Area.smooth.AGR"
anomalous <- anomPlot(longi.0255.dat, response=response, x="xDays+35.42857143", 
                      lower=2.5, start.time=40, breaks=seq(28, 42, by=2),  
                      vertical.line=29, whichPrint=c("innerPlot"),
                      y.title=response)
subs <- subset(anomalous$data, Area.smooth.AGR.anom & Days==33)
if (nrow(subs) == 0)
{  cat("\n#### No anomalous data here\n\n")
} else
{ subs <- subs[order(subs["Smarthouse"],subs["Treatment.1"], subs[response]),]
  print(subs[c(cols.output, response)])
  anom.ID <- unique(c(anom.ID, subs$Snapshot.ID.Tag))
  outerPlot <- anomalous$outerPlot  + geom_text(data=subs,
                                                aes_string(x = "xDays+35.42857143", y = response, 
                                                           label="Snapshot.ID.Tag"), 
                                                size=3, hjust=0.7, vjust=0.5)
  print(outerPlot)
}

#'### RGR anomalies - plot without anomalous plants followed by plot of anomalous plants
#+ "01-0255-RGRanomalies"
response <- "Area.smooth.RGR"
anomalous <- anomPlot(longi.0255.dat, response=response, x="xDays+35.42857143", 
                      groupsFactor="Treatment.1", lower=c(0.08,0.01), start.time=40, 
                      vertical.line=29, breaks=seq(28, 42, by=2), 
                      whichPrint=c("innerPlot"), y.title=response)
subs <- subset(anomalous$data, Area.smooth.RGR.anom & Days==33)
if (nrow(subs) == 0)
{  cat("\n#### No anomalous data here\n\n")
} else
{ subs <- subs[order(subs["Smarthouse"],subs["Treatment.1"], subs[response]),]
  print(subs[c(cols.output, response)])
  anom.ID <- unique(c(anom.ID, subs$Snapshot.ID.Tag))
  outerPlot <- anomalous$outerPlot + geom_text(data=subs, 
                                               aes_string(x = "xDays+35.42857143", y = response, 
                                                          label="Snapshot.ID.Tag"), 
                                               size=3, hjust=0.5, vjust=0.5) 
  print(outerPlot)
}

#'### Plot of smoothed Total Area with tags that seem anomalous highlighted
#+ "01-0255-AreaReplot"
longi.0255.dat$anom <- longi.0255.dat$Snapshot.ID.Tag %in% anom.ID
plt <- longiPlot(data = longi.0255.dat, x="xDays+35.42857143", response = "Area.smooth", 
                 y.title = "Total area (1000 pixels)", 
                 colour.column="anom", colour.values=c("grey80", "purple"), 
                 alpha=0.5, printPlot=FALSE)
plt <- plt + geom_vline(xintercept=29, linetype="longdash", size=1) +
  scale_x_continuous(breaks=seq(28, 42, by=2))
if (length(anom.ID) > 0)
  plt <- plt +  geom_text(data=subset(longi.0255.dat, anom & Days == 31),
                          aes_string(x = "xDays+35.42857143", y = "Area.smooth", 
                                     label="Snapshot.ID.Tag"), 
                          size=3, hjust=0.7, vjust=0.5) 
print(plt)
if (any(longi.0255.dat$anom))
{ print(subset(longi.0255.dat, anom & Days == 33)[c(cols.output,"Area.smooth")])
} else
  cat("\n#### No anomalous data here\n\n")
