



###############
#EXTRA COLUMNS
###############

args <- commandArgs(trailingOnly=TRUE)
m<-read.csv(("Input.csv"), header=T, stringsAsFactors=FALSE)
m= na.omit(m) 
#y<-as.character(m$snpid);y
#nchary<-nchar(y)

#x2<-gregexpr("_", y, perl=TRUE)
#x2<-as.data.frame(x2)
#location of '_' in names
#x2_1<-as.vector(unlist(x2[1,]));View(x2_1)
#x2_2<-as.vector(unlist(x2[2,]));View(x2_2)

#chrom<-substring(y,x2_1+4,x2_2-1);chrom
#pos<-substring(y,x2_2+1,nchary);pos


str(m)
#m$chrom<-as.numeric(as.character(Chr))
#m$pos<-as.numeric(as.character(Pos))
m$p_mark<-as.numeric(as.character(m$p_mark))
m$p_marktrt<-as.numeric(as.character(m$p_marktrt))



##############
#FDR VALUE
##############
library(qqman)
library(qvalue)

# Determine q-value for each p-value and if it is significant at an FDR level of 0.05
qobj <- qvalue(m$p_mark,pi0.method="bootstrap",fdr.level=0.05)
qvalues1 <- data.frame(qobj$pvalue,qobj$qvalue,qobj$significant)
colnames(qvalues1) <- c( 'pvalue marker', 'qvalue marker', 'significantmarker')


qobj <- qvalue(m$p_marktrt,pi0.method="bootstrap",fdr.level=0.05)
qvalues2 <- data.frame(qobj$pvalue,qobj$qvalue,qobj$significant)
colnames(qvalues2) <- c( 'pvalue MarkerxTreat', 'qvalue MarkerxTreat', 'significantMarkerxTreat')



mergedtable_m <- data.frame(m,qvalues1,qvalues2,stringsAsFactors = FALSE)
#View(mergedtable_m)

###################
#EXPORT RESULTS
###################
#write.csv(mergedtable_m ,'FDR.csv')




###############
#BONFERRONI 
################




# Determine genomewide significance using Bonferroni's correction
a <- nrow(m)
b <- 0.05/a
head(b)
# take the snp values that are less then the bonferroni value (significant snps)
b1 <- subset(m, m$p_mark < b)
b2<- subset(m, m$p_marktrt <b)
#View(b1)
#View(b2)
#dim(b1)
#dim(b2)
#write.table(b1, file = "markplot.BON.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
#write.table(b2, file = "marktrtplot.BON.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")





###############
#MANHATTAN PLOT
################
snps2highlight <- b1

tiff('markplot.BON.tiff')
manhattan(m,snp='snpid',chr="Chr", p="p_mark", bp='Pos',logp=T, ylab="-log10(pvalue)",
suggestiveline=FALSE,highlight=snps2highlight$snpid,genomewideline=-log10(b),ylim = c(0, 10))
dev.off()
#dev.new()

snps2highlight_int <- b2

tiff('marktrtplot.BON.tiff')

manhattan(m,chr="Chr",snp='snpid', p="p_marktrt", bp='Pos',logp=T, ylab="-log10(pvalue)",
 suggestiveline=FALSE,highlight=snps2highlight_int$snpid,
genomewideline=-log10(b),ylim = c(0, 10))

dev.off()







