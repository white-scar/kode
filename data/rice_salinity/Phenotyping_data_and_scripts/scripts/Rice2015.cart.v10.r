#'# Script to create cart data
options(java.parameters = "-Xmx10g")
library(knitr)
library(imageData)
library(dae)
library(XLConnect)
opts_chunk$set("tidy" = FALSE) 

#'# Step 6: Form single-value plant responses in Snapshot.ID.Tag order.
#'

"create.cart" <- function(longi.dat, responses.image, responses.smooth)
{ #'## 6a) Set up a data frame with factors only
  #+
  cart.dat <- longi.dat[longi.dat$Days == 31, 
                        c("Smarthouse","Lane","Position","Snapshot.ID.Tag","xPosn","xMainPosn",
                          "Zones","xZones","SHZones","ZLane","ZMainplots", "Subplots",
                          "Genotype.ID","Treatment.1")]
  cart.dat <- cart.dat[do.call(order, cart.dat), ]
  
  #'## 6b) Get responses based on first and last date.
  #'
  #'### Observation for first and last date
  cart.dat <- cbind(cart.dat, getDates(responses.image, data = longi.dat, 
                                       which.times = c(31), suffix = "first"))
  cart.dat <- cbind(cart.dat, getDates(responses.image, data = longi.dat, 
                                       which.times = c(42), suffix = "last"))
  cart.dat <- cbind(cart.dat, getDates(c("TUE.cum"), 
                                       data = longi.dat, 
                                       which.times = c(42), suffix = "last"))
  responses.smooth <- paste(responses.image, "smooth", sep=".")
  cart.dat <- cbind(cart.dat, getDates(responses.smooth, data = longi.dat, 
                                       which.times = c(31), suffix = "first"))
  cart.dat <- cbind(cart.dat, getDates(responses.smooth, data = longi.dat, 
                                       which.times = c(42), suffix = "last"))
  
  #'### Growth rates over whole period.
  #+
  tottime <- 42 - 31
  cart.dat <- within(cart.dat, 
                     { Area.AGR <- (Area.last - Area.first)/tottime
                       Area.RGR <- log(Area.last / Area.first)/tottime
                     })
  
  #'### Calculate water index over whole period
  cart.dat <- merge(cart.dat, 
                    intervalWUI("Area", water.use = "Transpiration", 
                                start.times = c(31), 
                                end.times = c(42), 
                                suffix = NULL, 
                                data = longi.dat, include.total.water = TRUE),
                    by = c("Snapshot.ID.Tag"))
  names(cart.dat)[match(c("Area.WUI","Transpiration.Total"),names(cart.dat))] <- c("Area.Overall.TUE", 
                                                                             "Transpiration.Overall")
  cart.dat$Transpiration.rate.Overall <- cart.dat$Transpiration.Overall / (42 - 31)
  
  #'## 6c) Add growth rates and water indices for intervals
  #'### Set up intervals
  #+
  start.days <- list(31,35,31,38)
  end.days <- list(35,38,38,42)
  suffices <- list("31to35","35to38","31to38","38to42")

  #'### Rates for specific intervals from the smoothed data by differencing
  #+
  for (r in responses.smooth)
  { for (k in 1:length(suffices))
    { cart.dat <- merge(cart.dat, 
                        intervalGRdiff(r, 
                                       which.rates = c("AGR","RGR"), 
                                       start.times = start.days[k][[1]], 
                                       end.times = end.days[k][[1]], 
                                       suffix.interval = suffices[k][[1]], 
                                       data = longi.dat),
                        by = "Snapshot.ID.Tag")
    }
  }
  
  #'### Water indices for specific intervals from the unsmoothed and smoothed data
  #+
  for (k in 1:length(suffices))
  { cart.dat <- merge(cart.dat, 
                      intervalWUI("Area", water.use = "Transpiration", 
                                  start.times = start.days[k][[1]], 
                                  end.times = end.days[k][[1]], 
                                  suffix = suffices[k][[1]], 
                                  data = longi.dat, include.total.water = TRUE),
                      by = "Snapshot.ID.Tag")
    names(cart.dat)[match(paste("Area.WUI", suffices[k][[1]], sep="."), names(cart.dat))] <- 
      paste("Area.TUE", suffices[k][[1]], sep=".")
    cart.dat[paste("Transpiration.rate", suffices[k][[1]], sep=".")] <- 
      cart.dat[[paste("Transpiration.Total", suffices[k][[1]], sep=".")]] / ( end.days[k][[1]] - start.days[k][[1]])
  }
  
  for (k in 1:length(suffices))
  { cart.dat <- merge(cart.dat, 
                      intervalWUI("Area.smooth", water.use = "Transpiration.smooth", 
                                  start.times = start.days[k][[1]], 
                                  end.times = end.days[k][[1]], 
                                  suffix = suffices[k][[1]], 
                                  data = longi.dat, include.total.water = TRUE),
                      by = "Snapshot.ID.Tag")
  names(cart.dat)[match(paste("Area.smooth.WUI", suffices[k][[1]], sep="."), names(cart.dat))] <- 
    paste("Area.smooth.TUE", suffices[k][[1]], sep=".")
  cart.dat[paste("Transpiration.smooth.rate", suffices[k][[1]], sep=".")] <- 
    cart.dat[[paste("Transpiration.smooth.Total", suffices[k][[1]], sep=".")]] / 
              ( end.days[k][[1]] - start.days[k][[1]])
  }
  cart.dat <- with(cart.dat, cart.dat[order(Snapshot.ID.Tag), ])
  return(cart.dat)
}


#'## Create cart.dat for 0254 and store it
cart.0254.dat <- create.cart(longi.0254.dat, responses.image, responses.smooth)
save(cart.0254.dat, file="02-cart.0254.dat.rda")
load(file="02-cart.0254.dat.rda")

#'## Create cart.dat for 0255 and store it
cart.0255.dat <- create.cart(longi.0255.dat, responses.image, responses.smooth)
save(cart.0255.dat, file="02-cart.0255.dat.rda")
load(file="02-cart.0255.dat.rda")
