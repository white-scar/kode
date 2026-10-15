#'# Script to investigate the fixed mixed model analyses of SIITs for 2015 Rice Salt Experiment
#'
#'## Set up
#+
library(dae)
library(asreml)
library(asremlPlus)
library(ggplot2)
library(QTLRel)
library(knitr)
library(XLConnect)
opts_chunk$set("tidy" = FALSE, "dev" = "png", "dpi" = 600, 
               "fig.width" = 9, "fig.height" = 9)


#'### Specify responses to be analyzed and their titles for plottng
#+
"gr.name" <- function(suffix, response, which.rate = "AGR")
{ name <- paste(response,which.rate,suffix[1],sep=".")
}
"gr.title" <- function(k, start.days, end.days, response, which.rate = "AGR", units)
{ title <- paste(response,which.rate,"Days",
                 paste(start.days[[k]], collapse="/"), "-",
                 paste(end.days[[k]], collapse="/"),
                 units, 
                 sep=" ")
  return(title)
}


responses.GR <- c("Area.smooth.AGR","Area.smooth.AGR","Area.smooth.RGR")
suffices.results <- c("diff", "SIIT", "SIIT")
responses.SIIT <- unlist(Map(paste, responses.GR, suffices.results, sep="."))
start.days <- list(31,35,31,38)
end.days <- list(35,38,38,42)
suffices <- list("31to35","35to38","31to38","38to42")
responses.SIITinterval <- as.vector(outer("Area.smooth.RGR", suffices, paste, sep="."))
responses.SIITinterval <- paste(responses.SIITinterval, "SIIT", sep=".")

#'### Set up titles for the responses
#+
titles <- c(lapply(1:length(start.days), gr.title, 
                   start.days = start.days, end.days = end.days, 
                   response = "Total Area", 
                   which.rate = "RGR", 
                   units = "(pixel/pixel/day) / (pixel/pixel/day)"))
names(titles) <- responses.SIITinterval
tran.power <- c(rep(1, length(responses.SIITinterval)))

#'## Function for fitting the fixed model to SIIT response[k]
fitFixedModelSIIT <- function(k, responses, titles, tran.power, df.supplied, SIIT.dat, BLUE.dat)
{ SIIT.dat <- subset(SIIT.dat, Genotype.ID != "Barke")
  #'Transform the data as specified by tran.power
  #+
  trans <- power.transform(var.name = responses[k], power = tran.power[k],
                           titles = titles, data = SIIT.dat)
  #'Select variable to analyse - if transformation, it is the last variable in data frame
  #+
  SIIT.dat <- trans$data
  titles <- trans$titles
  kanal <- match(trans$tvar.name, names(SIIT.dat))
  y.title <- titles[[trans$tvar.name]]
  
  #'Set up fixed model and fit initial model
  #+
  mod.ch <- paste(names(SIIT.dat)[kanal],
                  " ~ Smarthouse/(xZones + xMainPosn) + Genotype.ID")
  mod <- as.formula(mod.ch)
  cat("\n\n#### ",mod.ch,"\n\n")
  current.asr <- do.call(asreml, 
                         args=list(fixed = mod,
                                   random = ~ Smarthouse:(spl(xZones) + spl(xMainPosn)) + 
                                              Smarthouse:(dev(xZones) + dev(xMainPosn)),
                                   keep.order=TRUE, aom = TRUE,
                                   data = SIIT.dat, maxiter=50))
  current.asr <- update(current.asr)
  current.asrt <- asrtests(current.asr, NULL, NULL)
  
  #'remove boundaries, if any
  #+
  current.asrt <- rmboundary.asrtests(current.asrt)
  
  #'test deviations on Zones
  #+
  current.asrt <- testranfix.asrtests("Smarthouse:dev(xZones)", current.asrt, 
                                      pos=TRUE, denDF = "numeric")
  p <- current.asrt$test.summary$p[match("Smarthouse:dev(xZones)", current.asrt$test.summary$terms)]
  if (is.na(p))
    p <- 1
  if (p <= 0.05)
  { current.asrt <- addrm.terms.asrtests("Smarthouse:spl(xZones)", current.asrt, label="Smarthouse:spl(xZones)", 
                                         random=TRUE)
    df.supplied <- df.supplied[-(match("Smarthouse:xZones", rownames(current.asrt$wald.tab)))]
    current.asrt <- addrm.terms.asrtests("Smarthouse:xZones", current.asrt, label="Smarthouse:xZones")
  } else
  { #'test spline on Zones
    #+
    current.asrt <- testranfix.asrtests("Smarthouse:spl(xZones)", current.asrt, 
                                        pos=TRUE, denDF = "numeric")
  }
  
  #'test deviations on Positions
  #+
  current.asrt <- testranfix.asrtests("Smarthouse:dev(xMainPosn)", current.asrt, 
                                      pos=TRUE, denDF = "numeric")
  p <- current.asrt$test.summary$p[match("Smarthouse:dev(xMainPosn)", current.asrt$test.summary$terms)]
  if (is.na(p))
    p <- 1
  if (p <= 0.05)
  { current.asrt <- addrm.terms.asrtests("Smarthouse:spl(xMainPosn)", current.asrt, random=TRUE)
    df.supplied <- df.supplied[-(match("Smarthouse:xMainPosn", rownames(current.asrt$wald.tab)))]
    current.asrt <- addrm.terms.asrtests("Smarthouse:xMainPosn", current.asrt)
  } else
  {   #'test spline on Positions
    #+
    current.asrt <- testranfix.asrtests("Smarthouse:spl(xMainPosn)", current.asrt, 
                                        pos=TRUE, denDF = "numeric")
  } 
  
  #'Output results from final analysis
  #+
  if (current.asrt$asreml.obj$converge)
    cat("\n\n  Final model converged \n\n")
  else
    cat("\n\n  Final model failed to converged \n\n")
  print(summary(current.asrt$asreml.obj)$varcomp)
  cat("\n\n  Model-testing summary \n\n")
  print(current.asrt$test.summary)
  #Use supplied denominator df from the canonical analysis as an approximation
  wald.tab <- recalc.wald.tab.asrtests(current.asrt, denDF = "numeric", dDF.na = "supplied", 
                                       dDF.values = df.supplied)
  cat("\n\n  Pseudo-anova table for fixed terms \n\n")
  print(wald.tab)

  #'## Get predictions for Genotype.ID
  #+
  pred <- predict(current.asrt$asreml.obj, class = "Genotype.ID", 
                  sed = TRUE, workspace=32e6, pworkspace=32e6)
  pvals <- pred$predictions$pvals
  
  #'### Add BLUE and standard error to BLUE.dat
  geno.cols <- c("BLUE", "se")
  BLUE.dat <- merge(BLUE.dat, 
                    pvals[, c("Genotype.ID", "predicted.value","standard.error")], 
                    all.y=TRUE)
  geno.names <- sapply(geno.cols, 
                       function(col, response){name <- paste(response, col, sep=".")},
                       response = names(SIIT.dat)[kanal])
  names(BLUE.dat)[match(c("predicted.value","standard.error"),names(BLUE.dat))] <- geno.names
  
  #'Diagnostic plots of residuals
  #+ "02-diagnostics"
  resid.asr <- current.asrt$asreml.obj
  resid.asr <- update(resid.asr, aom=TRUE)
  SIIT.dat$resid <- residuals(resid.asr)
  SIIT.dat$fit <- fitted(resid.asr)
  res.dat <- SIIT.dat[!is.na(SIIT.dat$resid), ]
  repl <- table(SIIT.dat$Genotype.ID)
  if (any(repl == 1))
  { unrepl <- levels(SIIT.dat$Genotype.ID)[repl == 1]
    res.dat <- res.dat[!(res.dat$Genotype.ID %in% unrepl), ]
  }
  resfit.plot <-  ggplot(data=res.dat, aes(y = resid, x = fit)) +  
                  geom_point(na.rm=TRUE) +
                  labs(title = paste("Residuals versus fitted values for",
                                     names(res.dat)[kanal])) +
                  scale_x_continuous("Fitted values") + 
                  scale_y_continuous("Residuals")
  print(resfit.plot)
  qqPlot(res.dat$resid, 
         xlab = "Theoretical Quantiles", ylab = "Sample Quantiles", 
         main=paste("Normal probability plot of residuals for ", 
                    names(res.dat)[kanal]))
  #'## Save testing results and store summaries
  save <- vector(mode = "list", length = 0)
  save[["asreml.obj"]] <- current.asrt[["asreml.obj"]]
  save[["tests"]] <- current.asrt[["test.summary"]]
  save[["wald.tab"]] <- wald.tab
  save[["BLUEs"]] <- BLUE.dat
  invisible(save)
}


#'## Analyse responses for 0254
#'
#'### Loop over responses
#'
#+ "02-0254-SIITFixed"
nresp <- length(responses.SIITinterval)
save.0254.SIIT <- vector(mode = "list", length = nresp)
names(save.0254.SIIT) <- responses.SIITinterval
BLUE.0254.SIIT.dat <-fac.gen(list(Genotype.ID = levels(cart.0254.SIIT.dat$Genotype.ID)))
k <- 1
for (k in 1:nresp)
{ save.0254.SIIT[[responses.SIITinterval[k]]] <-
    fitFixedModelSIIT(k=k, responses.SIITinterval, titles, tran.power, 
                     df.supp=c(NA,9,rep(58, 3)), SIIT.dat=cart.0254.SIIT.dat, 
                     BLUE.0254.SIIT.dat)
  BLUE.0254.SIIT.dat <- save.0254.SIIT[[responses.SIITinterval[k]]][["BLUEs"]]
  save.0254.SIIT[[responses.SIITinterval[k]]][["BLUEs"]] <- NULL
}

#'### Save BLUEs
#+
excel.file <- file.path("Rice2015.0254.v8.xlsx")
wb = loadWorkbook(excel.file)
if (existsSheet(wb, "SIIT BLUEs"))
  removeSheet(wb, "SIIT BLUEs")
createSheet(wb, name="SIIT BLUEs")
writeWorksheet(wb, BLUE.0254.SIIT.dat, sheet="SIIT BLUEs")
nc <- ncol(BLUE.0254.dat)
setColumnWidth(wb, sheet = "SIIT BLUEs", column = 1, width = 3500)
setColumnWidth(wb, sheet = "SIIT BLUEs", column = 2:nc, width = 7500)
saveWorkbook(wb)


#'## Analyse responses for 0255
#'
#'### Loop over responses
#'
#+ "02-0255-SIITFixed"
nresp <- length(responses.SIITinterval)
save.0255.SIIT <- vector(mode = "list", length = nresp)
names(save.0255.SIIT) <- responses.SIITinterval
BLUE.0255.SIIT.dat <-fac.gen(list(Genotype.ID = levels(cart.0255.SIIT.dat$Genotype.ID)))
k <- 1
for (k in 1:nresp)
{ save.0255.SIIT[[responses.SIITinterval[k]]] <-
    fitFixedModelSIIT(k=k, responses.SIITinterval, titles, tran.power, 
                     df.supp=c(NA,9,rep(58, 3)), SIIT.dat=cart.0255.SIIT.dat, 
                     BLUE.0255.SIIT.dat)
  BLUE.0255.SIIT.dat <- save.0255.SIIT[[responses.SIITinterval[k]]][["BLUEs"]]
  save.0255.SIIT[[responses.SIITinterval[k]]][["BLUEs"]] <- NULL
}

#'### Save BLUEs
#+
excel.file <- file.path("Rice2015.0255.v8.xlsx")
wb = loadWorkbook(excel.file)
if (existsSheet(wb, "SIIT BLUEs"))
  removeSheet(wb, "SIIT BLUEs")
createSheet(wb, name="SIIT BLUEs")
writeWorksheet(wb, BLUE.0255.SIIT.dat, sheet="SIIT BLUEs")
nc <- ncol(BLUE.0255.dat)
setColumnWidth(wb, sheet = "SIIT BLUEs", column = 1, width = 3500)
setColumnWidth(wb, sheet = "SIIT BLUEs", column = 2:nc, width = 7500)
saveWorkbook(wb)
