#'# Script to do plots for cart data from the 2015 Rice experiments
#+

#'### Set up
#+
library(knitr)
library(ggplot2)
library(GGally)
library(Hmisc)
opts_chunk$set("tidy" = FALSE, "dev" = "png", "dpi" = 600, 
               "fig.width" = 9, "fig.height" = 9)
#              "out.width" = 1200, "out.height" = 1200) 


#'### Specify growth rates for plottng
"gr.name" <- function(suffix, response, which.rate = "AGR")
{ name <- paste(response,which.rate,suffix[1],sep=".")
}
suffices <- list("31to35","35to38","31to38","38to42")
responses <- c(sapply(suffices[1:length(suffices)],gr.name,
                      response = "Area.smooth", which.rate = "AGR"),
               sapply(suffices[1:length(suffices)],gr.name,
                      response = "Area.smooth", which.rate = "RGR"))

#This function puts text down the diagonal using my version of ggally_text
#The main modification is to the theme to remove the grid and ticks and to add a rectangle
my_ggally_text <- function (label, mapping = ggplot2::aes(color = "black"), xP = 0.5, 
                            yP = 0.5, xrange = c(0, 1), yrange = c(0, 1), ...) 
{ p <- ggplot() + xlim(xrange) + ylim(yrange) + 
       theme(legend.position = "none", 
             panel.background = element_blank(), 
             panel.grid.major = element_blank(), 
             axis.ticks = element_blank(), 
             panel.border = element_rect(linetype = "dashed", 
                                         colour = "black", 
                                         fill = NA)) +
       labs(x = NULL, y = NULL)
  new_mapping <- aes_string(x = xP * diff(xrange) + min(xrange, na.rm = TRUE), 
                            y = yP * diff(yrange) + min(yrange, na.rm = TRUE))
  if (is.null(mapping)) {
    mapping <- new_mapping
  }
  else {
     mapping$x <- new_mapping$x
     mapping$y <- new_mapping$y
     #    mapping <- addAndOverwriteAes(mapping, new_mapping)
  }
  colour <- as.character(mapping$colour)
  if (is.null(colour) || length(colour) < 1) 
    colour <- "grey50"
  mapping$colour <- NULL
  p <- p + geom_text(label = label, mapping = mapping, colour = colour, 
                     ...) + theme(legend.position = "none")
  p
}

my_ggpairs <- function(data, columns=1:ncol(data), labels, title = "")
{ nvars <- length(labels)
  if (nvars != length(columns))
    stop("Number of columns and labels must be the same")
  theme_set(theme_bw())
  gg1 <- ggpairs(data[,columns], title =title,
                 axisLabels="show",  columnLabels= rep(" ", nvars), 
                 diag=list(continuous="blank", discrete="blank")) 
  for (i in 1:nvars)
  { gtx <- my_ggally_text(labels[i], mapping = aes(col = "grey50"))
    gg1 <- putPlot(gg1, gtx, i, i)
  }
  gg1
}

#Function to plot data for each Smarthouse-Treatment combination
shBYtrtPlot <- function(cart.dat, responses, labels, title)
{ for (sh in levels(cart.dat$Smarthouse))
  { for(trt in levels(cart.dat$Treatment.1))
    { d <- subset(cart.dat, (Smarthouse==sh & Treatment.1==trt), select=responses)
      d <- na.omit(d)
      ktitle <- paste(title, ": ", sh, ", ", trt, sep="")
      if (nrow(d) != 0)
        print(my_ggpairs(d, labels=labels, title =ktitle))
    }
  }
}

shBYtrtcorr <- function(cart.dat, responses)
{ r <- vector("numeric", length=4)
  k <- 0
  for (sh in levels(cart.dat$Smarthouse))
  { for(trt in levels(cart.dat$Treatment.1))
    { k <- k + 1
      d <- subset(cart.dat, (Smarthouse==sh & Treatment.1==trt), select=responses)
      d <- na.omit(d)
      r[k] <- rcorr(d[[responses[2]]], d[[responses[6]]])[[1]][1,2]
    }
  }
  return(r)
}

#'### Set up labels
labs.AGR <- sapply(list("last","31 to 35","35 to 38","31 to 38","38 to 42"),
               function(suffix, response)
                 { name <- paste(response,suffix[1],sep="\n")},
               response = "Area.smooth")
labs.RGR <- sapply(list("31 to 35","35 to 38","31 to 38","38 to 42"),
                   function(suffix, response)
                   { name <- paste(response,suffix[1],sep="\n")},
                   response = "Area.smooth")


#'## Plot all plants
#+ "02-0254-GRatesPairsAll"
print(my_ggpairs(na.omit(cart.0254.dat[c("Area.last",responses[1:4])]), 
                 labels=labs.AGR, title ="Smooth AGR for Total area in 0254"))
print(my_ggpairs(na.omit(cart.0254.dat[c(responses[5:8])]), 
                 labels=labs.RGR, title ="Smooth RGR for Total area in 0254"))

#'## Plot TUE for all plants
#+ "02-0254-TUEPairsAll"
TUE <- c(sapply(suffices[1:length(suffices)],gr.name,
                response = "Area", which.rate = "TUE"),
         sapply(suffices[1:length(suffices)],gr.name,
                response = "Area.smooth", which.rate = "TUE"))
print(my_ggpairs(na.omit(cart.0254.dat[TUE]), 
                 labels=c(sapply(list("31 to 35","35 to 38","31 to 38","38 to 42"),
                                 function(suffix, response)
                                 { name <- paste(response,suffix[1],sep="\n")},
                                 response = "Area"),
                          labs.RGR), title ="TUE for unsmoothed and smoothed Total area in 0254"))

#+ "02-0255-GRatesPairsAll"
print(my_ggpairs(na.omit(cart.0255.dat[c("Area.last",responses[1:4])]), 
                 labels=labs.AGR, title ="Smooth AGR for Total area in 0255"))
print(my_ggpairs(na.omit(cart.0255.dat[c(responses[5:8])]), 
                 labels=labs.RGR, title ="Smooth RGR for Total area in 0255"))

#'## Plot TUE for all plants
#+ "02-0255-TUEPairsAll"
print(my_ggpairs(na.omit(cart.0255.dat[TUE]), 
                 labels=c(sapply(list("31 to 35","35 to 38","31 to 38","38 to 42"),
                                 function(suffix, response)
                                 { name <- paste(response,suffix[1],sep="\n")},
                                 response = "Area"),
                          labs.RGR), title ="TUE for unsmoothed and smoothed Total area in 0255"))

#'## Plots for each Smarthouse x Treatment combination
#+ "02-0254-GRatesPairsSxT"
shBYtrtPlot(cart.0254.dat, c("Area.last",responses[1:4]), 
            labs.AGR, "Smooth AGR for Total area in 0254")
shBYtrtPlot(cart.0254.dat, c(responses[5:8]), 
            labs.RGR, "Smooth RGR for Total area in 0254")

#+ "02-0255-GRatesPairsSxT"
shBYtrtPlot(cart.0255.dat, c("Area.last",responses[1:4]), 
            labs.AGR, "Smooth AGR for Total area in 0255")
shBYtrtPlot(cart.0255.dat, c(responses[5:8]), 
            labs.RGR, "Smooth RGR for Total area in 0255")
