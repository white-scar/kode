

###READ IN DATA

#######################
#TRAITS
#######################

pheno_ausind_sc<-read.csv('Master-File-Asreml-Nopanel_stacked_ausind.csv',header=T)
pheno_ausind_sc$treat<-as.factor(pheno_ausind_sc$treat)
pheno_ausind_sc$Taxa<-factor(pheno_ausind_sc$Taxa)

#####################
#KINSHIP MATRIX
#####################

r.df<-read.table('kinship_ausind.txt',header=T,row.names=1)
r.mat<-as.matrix(r.df)
rTaxa<-rownames(r.mat)

#FORMAT FOR ASREML
#create the inverse of the kinship matrix
r.mat.inv<-solve(r.mat)

#create a new factor IDnew
fac<-as.character(pheno_ausind_sc$Taxa)
pheno_ausind_sc$IDnew<-factor(fac,levels=rTaxa)


#create the inverse file in correct format for asreml
dimnames(r.mat.inv)<-list(rTaxa,rTaxa)
r.mat.inv.dinv<-data.frame(Column=as.vector(row(r.mat.inv)[!lower.tri(r.mat.inv)]),
Row=as.vector(col(r.mat.inv)[!lower.tri(r.mat.inv)]),
value=as.vector(r.mat.inv[!lower.tri(r.mat.inv)]))
attr(r.mat.inv.dinv,'rowNames')<-rTaxa


#############
#SNP DATA
#############

snp.mat<-read.table('snp_ausind.txt', header = T, row.names=1,as.is=T)






###LOOP


library(asreml)



#set up the traits to analyse before running the loop
#this is all traits
#if you just want to run one trait at a time then just choose the column in the data set relating to that trait
resp<-names(pheno_ausind_sc)[2:13]


#########
#FIT LOOP
#########


##########
#TRAITS j
##########


#this will do all traits in resp
for(j in 1:length(resp))
{#for j


yy<-which(names(pheno_ausind_sc)==resp[j])
trait<-pheno_ausind_sc[,yy]
traitname<-names(pheno_ausind_sc)[yy]

############
#CREATE THE VARIABLES TO CAPTURE INFORMATION FROM MODELS
############
#marker name
snpid<-rep(NA,dim(snp.mat)[1])

#pvalues
p_trt<-rep(NA,dim(snp.mat)[1])
p_mark<-rep(NA,dim(snp.mat)[1])
p_marktrt<-rep(NA,dim(snp.mat)[1])

p_pc1<-rep(NA,dim(snp.mat)[1])
p_pc2<-rep(NA,dim(snp.mat)[1])
p_pc3<-rep(NA,dim(snp.mat)[1])


#variance components
vc_m1<-rep(NA,dim(snp.mat)[1])
vc_Taxa<-rep(NA,dim(snp.mat)[1])
vc_resid<-rep(NA,dim(snp.mat)[1])

#marker and treat effects
eff_marktrt<-rep(NA,dim(snp.mat)[1])
eff_mark<-rep(NA,dim(snp.mat)[1])
eff_trt<-rep(NA,dim(snp.mat)[1])
eff_intercept<-rep(NA,dim(snp.mat)[1])

eff_marktrt_h<-rep(NA,dim(snp.mat)[1])
eff_mark_h<-rep(NA,dim(snp.mat)[1])

eff_pc1<-rep(NA,dim(snp.mat)[1])
eff_pc2<-rep(NA,dim(snp.mat)[1])
eff_pc3<-rep(NA,dim(snp.mat)[1])

#convergence and loglikelihood
conv<-rep(NA,dim(snp.mat)[1])
loglik<-rep(NA,dim(snp.mat)[1])


cat(traitname,'trait iteration',j,'\n')
sink(paste(traitname,'txt',sep='.'))
##################
#MARKERS WITHIN j
##################


# this will do all snps at once
for(i in 1:(dim(snp.mat)[1]))
{

cat(traitname,'trait iteration',j,'\n')
cat('iteration',i,'\n')

#marker name
snpid[i]<-rownames(snp.mat[i,])

#marker for analysis
marker<-as.vector(c(unlist(snp.mat[i,]),unlist(snp.mat[i,])))

#fit model
m1 <-asreml(trait~1+PC1+PC2+PC3+treat*factor(marker),random=~Taxa,
ginverse=list(Taxa=r.mat.inv.dinv),
 na.method.X='include', na.method.Y='include',
data=pheno_ausind_sc)

#pvalues
w_m1<-wald(m1)
p_pc1[i]<- w_m1[2,4]
p_pc2[i]<- w_m1[3,4]
p_pc3[i]<- w_m1[4,4]
p_trt[i]<- w_m1[5,4]
p_mark[i]<- w_m1[6,4]
p_marktrt[i]<- w_m1[7,4]

#variance components
vc_m1<-summary(m1,all=T)$varcomp
vc_Taxa[i]<-vc_m1[1,2]
vc_resid[i]<-vc_m1[2,2]

#marker and treat effects
sm<-summary(m1,all=T)$coef.fixed
#if not heterozygous
if(length(sm)==36 )
{
eff_marktrt[i]<-sm[[4]]
eff_mark[i]<-sm[[6]]
eff_trt[i]<-sm[[8]]
eff_pc1[i]<-sm[[9]]
eff_pc2[i]<-sm[[10]]
eff_pc3[i]<-sm[[11]]
eff_intercept[i]<-sm[[12]]

} else { #if heterozygous
eff_marktrt_h[i]<-sm[[5]]
eff_marktrt[i]<-sm[[6]]
eff_mark_h[i]<-sm[[8]]
eff_mark[i]<-sm[[9]]
eff_trt[i]<-sm[[11]]

eff_pc3[i]<-sm[[12]]
eff_pc2[i]<-sm[[13]]
eff_pc1[i]<-sm[[14]]

eff_intercept[i]<-sm[[15]]
}


conv[i]<-m1$last.message
loglik[i]<-m1$loglik

#this will periodically save results which guards against losing things if crashes
#this address will need to be changed to wherever you are saving the results
#also add iterations where you want it to be saved
if(i==5000|i==7500|i==10000|i==15000|i==300000|i==400000|i==500000|i==600000|i==dim(snp.mat)[1])
{
#if you have a separate folder set up for each response then this will save the .RData into this folder
#hh<-paste('Run_image_backup\\',traitname,'\\','.RData',sep='')
#otherwise each time its run for a different variable results will be overwritten
#hh<-"Run_image_backup.RData"
#save.image(hh)

}

}#end i

#save results in temp object
temp<-as.data.frame(cbind(snpid,p_trt,p_mark,p_marktrt,p_pc1,p_pc2,p_pc3,vc_Taxa,vc_resid,
eff_intercept,eff_trt,eff_mark,eff_marktrt,eff_marktrt_h,eff_mark_h,eff_pc1,eff_pc2,eff_pc3,conv,loglik))

#write results to txt file
write.csv(temp,paste(traitname,'csv',sep='.'),row.names=F)

rm(snpid,p_trt,p_mark,p_marktrt,p_pc1,p_pc2,p_pc3,vc_Taxa,vc_resid,
eff_intercept,eff_trt,eff_mark,eff_marktrt,eff_marktrt_h,eff_mark_h,eff_pc1,eff_pc2,eff_pc3,conv,loglik)
sink()

}#end j

