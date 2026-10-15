#'# Script to perform the fixed mixed model analyses of the growth rates for 2015 Rice Salt Experiment
#'
#'## Set up
#+
options(java.parameters = "-Xmx10g", width=100)
library(knitr)
library(dae)
library(asreml)
library(asremlPlus)
library(ggplot2)
library(QTLRel)
library(XLConnect)
opts_chunk$set("tidy" = FALSE, "dev" = "png", "dpi" = 600, 
               "fig.width" = 9, "fig.height" = 9)


#'### Specify responses to be analyzed and their titles for plottng
#+
"gr.name" <- function(suffix, response, which.rate = "AGR")
{ if (is.null(which.rate))
    name <- paste(response,suffix[1],sep=".")
  else
    name <- paste(response,which.rate,suffix[1],sep=".")
  return(name)
}
"gr.title" <- function(k, start.days, end.days, response, which.rate = "AGR", units)
{ if (is.null(which.rate))
    title <- paste(response,"Days",
                   paste(start.days[[k]], collapse="/"), "-",
                   paste(end.days[[k]], collapse="/"),
                   units, 
                   sep=" ")
  else
    title <- paste(response,which.rate,"Days",
                   paste(start.days[[k]], collapse="/"), "-",
                   paste(end.days[[k]], collapse="/"),
                   units, 
                   sep=" ")
  return(title)
}
start.days <- list(31,35,31,38)
end.days <- list(35,38,38,42)
suffices <- list("31to35","35to38","31to38","38to42")
responses <- c(sapply(suffices[1:length(suffices)],gr.name,
                      response = "Area.smooth", which.rate = "AGR"),
               sapply(suffices[1:length(suffices)],gr.name,
                      response = "Area.smooth", which.rate = "RGR"),
               sapply(suffices[1:length(suffices)],gr.name,
                      response = "Area.smooth.TUE", which.rate = NULL),
               sapply(suffices[1:length(suffices)],gr.name,
                      response = "Transpiration.smooth.rate", which.rate = NULL))
#'### Set up titles for the responses
#+
titles <- c(lapply(1:length(start.days), gr.title, 
                   start.days = start.days, end.days = end.days, 
                   response = "Total Area", 
                   which.rate = "AGR", 
                   units = "(1000 pixels / day)"),
            lapply(1:length(start.days), gr.title, 
                   start.days = start.days, end.days = end.days, 
                   response = "Total Area", 
                   which.rate = "RGR", 
                   units = "(pixel / pixel / day)"),
            lapply(1:length(start.days), gr.title, 
                   start.days = start.days, end.days = end.days, 
                   response = "Water use index", 
                   which.rate = NULL, 
                   units = "(pixel / mL)"),
            lapply(1:length(start.days), gr.title, 
                   start.days = start.days, end.days = end.days, 
                   response = "Transpiration rate", 
                   which.rate = NULL, 
                   units = "(pixel / mL / day)"))
names(titles) <- responses
tran.power <- c(rep(1, length(responses)))

#'## Function for fitting the fixed model to response[k]
fitFixedModel <- function(k, responses, titles, tran.power, df.supplied, cart.dat, BLUE.dat)
{ cart.dat <- subset(cart.dat, Genotype.ID != "Barke")
  #'Transform the data as specified by tran.power
  #+
  trans <- power.transform(var.name = responses[k], power = tran.power[k],
                           titles = titles, data = cart.dat)
  #'Select variable to analyse - if transformation, it is the last variable in data frame
  #+
  cart.dat <- trans$data
  titles <- trans$titles
  kanal <- match(trans$tvar.name, names(cart.dat))
  y.title <- titles[[trans$tvar.name]]
  
  #'Set up fixed model and fit initial model
  #+
  mod.ch <- paste(names(cart.dat)[kanal],
                  " ~ Smarthouse/(xZones + xMainPosn) + Genotype.ID*Treatment.1")
  mod <- as.formula(mod.ch)
  cat("\n\n#### ",mod.ch,"\n\n")
  current.asr <- do.call(asreml, 
                         args=list(fixed = mod,
                                   random = ~ Smarthouse:(spl(xZones) + spl(xMainPosn)) + 
                                              Smarthouse:(dev(xZones) + dev(xMainPosn)) + 
                                              SHZones:ZMainplots,
                                   rcov = ~ at(Treatment.1):SHZones:ZMainplots,
                                   keep.order=TRUE, data = cart.dat, maxiter=50))
  current.asrt <- asrtests(current.asr, NULL, NULL)
  
  #'remove boundaries, if any
  #+
  current.asrt <- rmboundary.asrtests(current.asrt)
  
  #'If still a boundary, then remove Mainplots
  #+
  if ("Boundary" %in% summary(current.asrt$asreml.obj)$varcomp$constraint)
    current.asrt <- addrm.terms.asrtests("SHZones:ZMainplots", current.asrt, random=TRUE, 
                                         label="SHZones:ZMainplots", update=FALSE)

  #'test unequal Treatment.1 residual variances
  #+
  current.asrt <- testrcov.asrtests("Treatment.1:SHZones:ZMainplots", current.asrt, 
                                    simple=TRUE, label="Unequal Treatment.1 Residual", 
                                    denDF = "numeric")
  
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
  { current.asrt <- addrm.terms.asrtests("Smarthouse:spl(xMainPosn)", current.asrt, label="Smarthouse:spl(xMainPosn)", 
                                         random=TRUE)
    df.supplied <- df.supplied[-(match("Smarthouse:xMainPosn", rownames(current.asrt$wald.tab)))]
    current.asrt <- addrm.terms.asrtests("Smarthouse:xMainPosn", current.asrt, label="Smarthouse:xMainPosn")
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

  #'## Get predictions for Genotype.ID:Treatment.1
  #+
  pred <- predict(current.asrt$asreml.obj, class = "Treatment.1:Genotype.ID", 
                  sed = TRUE, workspace=32e6, pworkspace=32e6)
  pvals <- pred$predictions$pvals
  
  #'### Add BLUE and standard error to BLUE.dat
  geno.cols <- c("BLUE", "se")
  BLUE.dat <- merge(BLUE.dat, 
                    pvals[, c("Genotype.ID", "Treatment.1", "predicted.value","standard.error")], 
                    all.y=TRUE)
  geno.names <- sapply(geno.cols, 
                       function(col, response){name <- paste(response, col, sep=".")},
                       response = names(cart.dat)[kanal])
  names(BLUE.dat)[match(c("predicted.value","standard.error"),names(BLUE.dat))] <- geno.names
  
  #'Diagnostic plots of residuals
  #+ "02-diagnostics"
  resid.asr <- current.asrt$asreml.obj
  resid.asr <- update(resid.asr, aom=TRUE)
  cart.dat$resid <- residuals(resid.asr)
  cart.dat$fit <- fitted(resid.asr)
  res.dat <- cart.dat[!is.na(cart.dat$resid), ]
  repl <- table(cart.dat$Genotype.ID)/2
  if (any(repl == 1))
  { unrepl <- levels(cart.dat$Genotype.ID)[repl == 1]
    res.dat <- res.dat[!(res.dat$Genotype.ID %in% unrepl), ]
  }
  resfit.plot <-  ggplot(data=res.dat, 
                         aes(y = resid, x = fit)) +  
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
  resfit.plot <-  ggplot(data=res.dat, 
                         aes(y = resid, x = fit)) +  
                  geom_point(colour = "blue", na.rm=TRUE) +
                  facet_wrap(~ Treatment.1) + 
                  labs(title = paste("Residuals versus fitted values by Treatment for",
                                     names(res.dat)[kanal])) +
                  scale_x_continuous("Fitted values") + 
                  scale_y_continuous("Residuals") + 
                  theme(strip.text.x = element_text(size=10))
  print(resfit.plot)
  qq.plot <-  ggplot(data=res.dat, 
                     aes(sample = resid)) + 
              stat_qq(geom = "point", na.rm=TRUE) + 
              facet_wrap(~ Treatment.1) +
              geom_abline(slope = 1) + 
              labs(title = paste("Normal probablity plot by Treatment for ",
                                 names(res.dat)[kanal])) +
              scale_x_continuous("Theoretical quantiles") + 
              scale_y_continuous("Sample quantiles") + 
              theme(strip.text.x = element_text(size=10))
  print(qq.plot)

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
#'## Load data and re-order  for analysis
#+
cart.0254.dat <- with(cart.0254.dat, cart.0254.dat[order(Treatment.1:SHZones:ZMainplots),])

#'Check properties of the design as observed by performing a canonical analysis
set.daeTolerance(1e-04, 1e-06)
Rice2015.0254.canon <- projs.canon(list(cart = ~ Smarthouse/Zones/ZMainplots/Subplots,
                                     trt= ~ Smarthouse/(xMainPosn + xZones) + Genotype.ID*Treatment.1), 
                                orthog="eigen", keep.order=TRUE,
                                data=cart.0254.dat)
summary(Rice2015.0254.canon)

#'### Loop over responses
#'
#+ "02-0254-GRatesFixed"
nresp <- length(responses)
save.0254 <- vector(mode = "list", length = nresp)
names(save.0254) <- responses
BLUE.0254.dat <-fac.gen(list(Treatment.1 = levels(cart.0254.dat$Treatment.1),
                             Genotype.ID = levels(cart.0254.dat$Genotype.ID)))
k <- 3
for (k in 1:nresp)
{ save.0254[[responses[k]]] <-fitFixedModel(k=k, responses, titles, tran.power, 
                                df.supp=c(NA,9,rep(c(58,69), c(3,2))), 
                                cart.dat=cart.0254.dat, BLUE.0254.dat)
  BLUE.0254.dat <- save.0254[[responses[k]]][["BLUEs"]]
  save.0254[[responses[k]]][["BLUEs"]] <- NULL
}
  
#'### Save BLUEs
#+
excel.file <- file.path("Rice2015.0254.v8.xlsx")
wb = loadWorkbook(excel.file)
if (existsSheet(wb, "Cart BLUEs"))
  removeSheet(wb, "Cart BLUEs")
createSheet(wb, name="Cart BLUEs")
writeWorksheet(wb, BLUE.0254.dat, sheet="Cart BLUEs")
nc <- ncol(BLUE.0254.dat)
setColumnWidth(wb, sheet = "Cart BLUEs", column = 1:2, width = 3500)
setColumnWidth(wb, sheet = "Cart BLUEs", column = 3:nc, width = 5000)
saveWorkbook(wb)


#'## Analyse responses for 0255
#'
#'## Load data and re-order  for analysis
#+
cart.0255.dat <- with(cart.0255.dat, cart.0255.dat[order(Treatment.1:SHZones:ZMainplots),])

#'Check properties of the design as observed by performing a canonical analysis
Rice2015.0255.canon <- projs.canon(list(cart = ~ Smarthouse/Zones/ZMainplots/Subplots,
                                        trt= ~ Smarthouse/(xMainPosn + xZones) + Genotype.ID*Treatment.1), 
                                   orthog="eigen", keep.order=TRUE,
                                   data=subset(cart.0255.dat, Genotype.ID != "Barke"))
summary(Rice2015.0255.canon)

#'### Loop over responses
#'
#+ "02-0255-GRatesFixed"
nresp <- length(responses)
save.0255 <- vector(mode = "list", length = nresp)
names(save.0255) <- responses
BLUE.0255.dat <-fac.gen(list(Treatment.1 = levels(cart.0255.dat$Treatment.1),
                             Genotype.ID = levels(cart.0255.dat$Genotype.ID)))
for (k in 1:nresp)
{ save.0255[[responses[k]]] <-fitFixedModel(k=k, responses, titles, tran.power, 
                                            df.supp=c(NA,9,rep(c(58,69), c(3,2))), 
                                            cart.dat=cart.0255.dat, BLUE.0255.dat)
  BLUE.0255.dat <- save.0255[[responses[k]]][["BLUEs"]]
  save.0255[[responses[k]]][["BLUEs"]] <- NULL
}

#'### Save BLUEs
#+
excel.file <- file.path("Rice2015.0255.v8.xlsx")
wb = loadWorkbook(excel.file)
if (existsSheet(wb, "Cart BLUEs"))
  removeSheet(wb, "Cart BLUEs")
createSheet(wb, name="Cart BLUEs")
writeWorksheet(wb, BLUE.0255.dat, sheet="Cart BLUEs")
nc <- ncol(BLUE.0255.dat)
setColumnWidth(wb, sheet = "Cart BLUEs", column = 1:2, width = 3500)
setColumnWidth(wb, sheet = "Cart BLUEs", column = 3:nc, width = 5000)
saveWorkbook(wb)
