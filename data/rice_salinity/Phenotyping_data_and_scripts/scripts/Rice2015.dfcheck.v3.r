#'# Script to look at different values of the df
#'
#'

#'## Values of df for which to obtain plots
df <- c(5,7,4,9)
vline <- list(geom_vline(xintercept=29, linetype="longdash", size=1),
              scale_x_continuous(breaks=seq(28, 42, by=2)))

#'## Obtain plots for 0254
#+ "01-0254-df"
probeDF(longi.0254.dat, df=df, x="xDays+35.42857143", ggplotFuncs=vline, 
        which.plots=c("response","AGR","RGR"), na.rm=TRUE)

#'## Obtain plots for 0255
#+ "01-0255-df"
probeDF(longi.0255.dat, df=df, x="xDays+35.42857143", ggplotFuncs=vline, 
        which.plots=c("response","AGR","RGR"), na.rm=TRUE)
