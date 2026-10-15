#FILE THAT AMENDS AND EXPORTS DATA SO READY FOR IMPORT


#####################
#KINSHIP MATRIX
#####################

r.df<-read.table('Kin_IND700.maf005.nomissing.txt',header=F,skip=1)


dim(r.df)
View(r.df)
#make matrix
#put an X infront of numeric taxa name
r.df$V1<-paste('X',r.df$V1,sep='')
is.numeric(r.df$V2)

rTaxa<-as.character(r.df[,1]);rTaxa

r.mat<-as.matrix(r.df[,-1])
rownames(r.mat)<-rTaxa
colnames(r.mat)<-rTaxa


#check solvable
head(solve(r.mat))
dim(r.mat)

#######################
#TRAITS
#######################

pheno<-read.csv('../Master-File-Asreml-Nopanel.csv',header=T)


#change name to Taxa so consistent across files
pheno$Taxa<-as.character(pheno$Taxa)
#put an X infront of numeric taxa name
pheno$Taxa<-paste('X',pheno$Taxa,sep='')
pheno$Taxa<-factor(pheno$Taxa)

pheno_ind<-pheno

#############################################################
#AMEND TRAIT FILE (pheno_ind) TO JUST CONTAIN WHATS IN THE KINSHIP MATRIX
#############################################################

mm<-match(rownames(r.mat),unique(as.character(pheno_ind$Taxa)))
mm[which(!is.na(mm))]
pheno_ind<-pheno_ind[mm[which(!is.na(mm))],]
dim(pheno_ind)

######################################
#AMEND KINSHIP MATRIX TO MATCH PHENO
######################################

zz<-match(rownames(r.mat),pheno_ind$Taxa);zz
which(is.na(zz))

#r.mat<-r.mat[-which(is.na(zz)),-which(is.na(zz))]
#dim(r.mat)

#check
match(unique(as.character(pheno_ind$Taxa)),rownames(r.mat))-1:dim(r.mat)[1]



#export file
write.table(r.mat,'kinship_ind.txt')


#######################
#STACKED PHENO FILE
#######################
head(pheno_ind)
names(pheno_ind)[2:25]
#create a file with control and salt stacked on each other
pheno_ind_contr<-pheno_ind[,c(1,2:5,10:13,18:21)]
dim(pheno_ind_contr)#228  13
names(pheno_ind_contr)
names(pheno_ind_contr)<-c( "Taxa" ,       "RGR.2to9" , "RGR.2to6" , "RGR.6to9", 
 "RGR.9to13", "TR.2to9" ,  "TR.2to6" ,  "TR.6to9" , 
 "TR.9to13",  "TUE.2to9" , "TUE.2to6" , "TUE.6to9" ,
 "TUE.9to13")


pheno_ind_salt<-pheno_ind[,c(1,6:9,14:17,22:25)]
dim(pheno_ind_salt)#228 13
names(pheno_ind_salt)
names(pheno_ind_salt)<-c(  "Taxa" ,   "RGR.2to9" , "RGR.2to6" , "RGR.6to9" ,
 "RGR.9to13", "TR.2to9" ,  "TR.2to6" ,  "TR.6to9" , 
 "TR.9to13" , "TUE.2to9" , "TUE.2to6" , "TUE.6to9" ,
 "TUE.9to13"
)

pheno_ind_sc<-rbind(pheno_ind_salt,pheno_ind_contr)

#create salt/control treatment
pheno_ind_sc$treat<-rep(c('s','c'),rep(dim(pheno_ind_salt)[1],2))
pheno_ind_sc$treat<-as.factor(pheno_ind_sc$treat)




#########
#PCA DATA
#########

pc<-read.table('PC_IND700.txt',header=T,skip=2)
head(pc)

pc$Taxa<-paste('X',pc$Taxa,sep='')

mm<-match(pc$Taxa,pheno_ind$Taxa,)

ww<-which(is.na(mm));ww

#pc<-pc[-ww,]
dim(pc)
dim(pheno_ind)

#MERGE PHENO AND PC FILE

pheno_ind_sc<-merge(pheno_ind_sc,pc, by='Taxa')


#EXPORT AMMENDED FILE
write.csv(pheno_ind_sc,'Master-File-Asreml-Nopanel_stacked_indica.csv',row.names=F)




#############
#SNP DATA
#############
#note download tassel v5 and convert vcf to hapmap format

snp<-read.table('Hapmap_IND700.maf005.nomissing.hmp.txt', header = T,as.is=T, comment.char = "")#,na.strings = "N")
dim(snp)
View(snp[1:20,1:20])

dim(snp)#19436    239

d1<-dim(snp)[1];d1
d2<-dim(snp)[2];d2
names(snp)[c(1,6)]<-c('rs' ,'assembly')

View(snp[1:5,1:20])
snp.mat<-snp[,12:(dim(snp)[2])]
#View(snp.mat)
rownames(snp.mat)<-snp$rs


nn<-match(names(snp.mat),pheno_ind$Taxa)
which(!is.na(nn))
snp.mat<-snp.mat[,which(!is.na(nn))]
dim(snp.mat)
dim(r.mat)
dim(r.mat)*2
dim(pheno_ind_sc)

write.table(snp.mat,'snp_indica.txt')


#check number of allele options in snp
l<-apply(snp.mat,1,function(x) length(table(unlist(x))))
table(l)
#   2     3 
# 5549 13887 
#which heterozygotes
head(which(l==2))
head(which(l==3))
head(which(l==4))
