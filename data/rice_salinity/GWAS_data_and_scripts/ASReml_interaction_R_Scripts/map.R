#add map details to results table
#################
#MAP
#################

snp<-read.table('Hapmap_AUS700.maf005.nomissing.hmp.txt', header = T,as.is=T, comment.char = "")#,na.strings = "N")
dim(snp)
#View(snp)

dim(snp)#16538 157
d1<-dim(snp)[1];d1
d2<-dim(snp)[2];d2
names(snp)[c(1,6)]<-c('rs' ,'assembly')


map<-snp[,1:11]
dim(map)
head(map)


results<-read.table('RGR.2to6.txt',header=T,as.is=T)
head(map)
head(results)
#make sure snps are in same order in map and results
head(match(results$snpid,map$rs))
mm<-match(results$snpid,map$rs)
#change order of snps in map to snps in results
map<-map[mm,]

head(map)
head(results)

results$rs<-map$rs
#double check
which(results$rs!=results$snpid)#should be integer(0)


results$chrom<-map$chrom
results$pos<-map$pos

