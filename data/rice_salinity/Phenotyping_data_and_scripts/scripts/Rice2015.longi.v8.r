#'# Steps 2, 3 & 4: Create longitudinal imaging responses and factors, including smoothed responses
library(knitr)
library(imageData)
library(dae)
opts_chunk$set("tidy" = FALSE) 

"create.longi" <- function(raw.dat, evap.xlsx, responses.image, responses.smooth)
{ #'# Step 2: Select imaging variables and add covariates and factors (produces longi.dat)
  longi.dat <- longitudinalPrime(data=raw.dat, 
                                 smarthouse.lev=c("NE","NW"))
  with(longi.dat, table(Genotype.ID,Reps))

  longi.dat <- designFactors(data=longi.dat, insertName = "xDays", designfactorMethod="StandardOrder")
  
  #'### Adjust the day 30 Water.Loss and Water.Amount for 100 ml added on day 29
  longi.dat$Water.Loss[longi.dat$Days=="30"] <- longi.dat$Water.Loss[longi.dat$Days=="30"] + 100
  longi.dat$Water.Amount[longi.dat$Days=="30"] <- longi.dat$Water.Amount[longi.dat$Days=="30"] + 100
  
  #'### Add evap carts for each plant cart
  evap <- readRDS("Evap.carts.rda")
  evap <- evap[evap$Cart.type=="P",]
  names(evap)[1:2] <- c("Lane","Position")
  longi.dat <- merge(longi.dat, evap[c("Lane","Position","Evap.Cart")], all.x=TRUE,sort=FALSE)
  
  #Get evaporation data
  evap.dat <- readWorksheetFromFile(file = evap.xlsx, sheet= "empty pots")
  evap.dat <- within(evap.dat, 
                     { Days <- factor(Time.after.Planting..d.)
                       Evaporation <-   unlist(by(Weight.After, list(Snapshot.ID.Tag), 
                                                  FUN=calcLagged)) - Weight.Before
                    })
  #Adjust day 30 for 100 ml added on day 29
  evap.dat$Evaporation[evap.dat$Days=="30"] <- evap.dat$Evaporation[evap.dat$Days=="30"] + 100

  #Get evaporation data into longi and calculate Plant Transpiration
  evap.dat$Evap.Cart <- rep(c(1,13,2,14,3,15,4,16,5,17,6,18,7,19,8,20,9,21,10,22,11,23,12,24), each=14, times=2)
  longi.dat <- merge(longi.dat, evap.dat[c("Smarthouse","Evap.Cart", "Days", "Evaporation")], 
                     all.x=TRUE, sort=FALSE)

  #Remove Day 40 from 0254
  if (grepl("0254", evap.xlsx, fixed=TRUE))
  { #Set each water.var on day 41 to the sum of that for days 40 and 41 and remove day 40
    longi.dat <- split(longi.dat, longi.dat$Days)
    longi.dat[["40"]] <- longi.dat[["40"]][order(longi.dat[["40"]]$Snapshot.ID.Tag),]
    longi.dat[["41"]] <- longi.dat[["41"]][order(longi.dat[["41"]]$Snapshot.ID.Tag),]
    water.vars <- c("Water.Amount", "Water.Loss", "Evaporation")
    longi.dat[["41"]][water.vars]  <- longi.dat[["41"]][water.vars] + longi.dat[["40"]][water.vars]
    longi.dat <- do.call(rbind, longi.dat)
    longi.dat <- with(longi.dat, longi.dat[Days != "40",])
    longi.dat$Days <- factor(longi.dat$Days)
  }

  #'## Particular edits to longi.dat
  longi.dat <- within(longi.dat, 
                      { Days.after.Salting <- as.numfac(Days) - 29
                        Transpiration <- Water.Loss - Evaporation
                      })
  longi.dat <- with(longi.dat, longi.dat[order(Snapshot.ID.Tag,Days), ])
  rownames(longi.dat) <- NULL
  
  
  #'# Step 3: Form derived traits that result in a value for each observation
  #+
  #'## Form growth rates, by differencing, for each observation of a subset of responses
  #+
  longi.dat <- splitContGRdiff(longi.dat, responses.image, 
                                 INDICES="Snapshot.ID.Tag",
                                 which.rates = c("AGR","RGR"))

  #'## Form Area.TUE
  longi.dat <- within(longi.dat, 
                      { Transpiration.rate <- Transpiration/Days.diffs
                        Area.TUE <- Area.AGR / Transpiration.rate
                      })

  #'## Add cumulative responses 
  longi.dat <- within(longi.dat, 
                      { Transpiration.Cum <- unlist(by(Transpiration, Snapshot.ID.Tag, 
                                                       cumulate, exclude.1st=TRUE))
                        TUE.cum <- Area / Transpiration.Cum 
                      })

  #'# Step 4: Fit splines to smooth the longitudinal trends in the primary traits and calculate their growth rates
  #'
  #'## Smooth responses
  #+
  for (response in c(responses.image, "Transpiration.rate"))
    longi.dat <- splitSplines(longi.dat, response, x="xDays", INDICES = "Snapshot.ID.Tag", 
                              df = 4, na.rm=TRUE)
  longi.dat <- with(longi.dat, longi.dat[order(Snapshot.ID.Tag, xDays), ])
  
  #'## Loop over smoothed responses, forming growth rates by differences
  #+
  responses.GR <- paste(responses.smooth, "AGR", sep=".")
  longi.dat <- splitContGRdiff(longi.dat, responses.smooth, 
                               INDICES="Snapshot.ID.Tag",
                               which.rates = c("AGR","RGR"))

  #'## Form Area.smooth.TUI and smooth Transpiration
  longi.dat <- within(longi.dat, 
                      { Area.smooth.TUE <- Area.smooth.AGR / Transpiration.rate.smooth
                        Transpiration.smooth <- Transpiration.rate.smooth * Days.diffs
                      })

  #'## Finalize longi.dat
  longi.dat <- with(longi.dat, longi.dat[order(Snapshot.ID.Tag, xDays), ])
  rownames(longi.dat) <- NULL
  return(longi.dat)
}


#'### Set responses
responses.image <- c("Area")
responses.smooth <- paste(responses.image, "smooth", sep=".")

#'## Create longi.dat for 0254 and store it
longi.0254.dat <- create.longi(raw.0254.dat, evap.xlsx="0254 evapdata.xlsx", 
                               responses.image, responses.smooth)
save(longi.0254.dat, file="01-longi.0254.dat.rda")
load(file="01-longi.0254.dat.rda")

#'## Create longi.dat for 0255 and store it
longi.0255.dat <- create.longi(raw.0255.dat, evap.xlsx="0255 evapdata.xlsx", 
                               responses.image, responses.smooth)
save(longi.0255.dat, file="01-longi.0255.dat.rda")
load(file="01-longi.0255.dat.rda")
