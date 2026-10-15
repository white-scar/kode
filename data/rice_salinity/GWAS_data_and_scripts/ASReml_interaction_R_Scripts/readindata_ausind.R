#FILE THAT AMENDS AND EXPORTS DATA SO READY FOR IMPORT 


#####################
#KINSHIP MATRIX
#####################

r.df<-read.table('Kin_INDAUS700.maf005.nomissing.txt',header=F,skip=1)


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
tail(eigen(r.mat)$values)

x<-solve(r.mat[c(1:278,280:315,317:374),c(1:278,280:315,317:374)])
rownames(r.mat)[c(279,316)]
which(rownames(r.mat)=='X77298')
which(rownames(r.mat)=='X121106')
which(rownames(r.mat)=='X32571')
which(rownames(r.mat)=='X117559')

head(r.mat[c(87,279,117,316),c(87,279,117,316)])

#first two have same kinship and last two have same kinship
#can see this if reorder in excel
#exclude 'X121106' and 'X117559'

temp<-r.mat[c(1:278,280:315,317:373),c(1:278,280:315,317:373)]
tail(eigen(temp)$values)

r.mat<-r.mat[c(1:278,280:315,317:373),c(1:278,280:315,317:373)]
dim(r.mat)

tail(eigen(r.mat)$values)


#######################
#TRAITS
#######################

pheno<-read.csv('../Master-File-Asreml-Nopanel.csv',header=T)


#change name to Taxa so consistent across files
pheno$Taxa<-as.character(pheno$Taxa)
#put an X infront of numeric taxa name
pheno$Taxa<-paste('X',pheno$Taxa,sep='')
pheno$Taxa<-factor(pheno$Taxa)

pheno_ausind<-pheno

#############################################################
#AMEND TRAIT FILE (pheno_ausind) TO JUST CONTAIN WHATS IN THE KINSHIP MATRIX
#############################################################

mm<-match(rownames(r.mat),unique(as.character(pheno_ausind$Taxa)))
mm[which(!is.na(mm))]
pheno_ausind<-pheno_ausind[mm[which(!is.na(mm))],]
dim(pheno_ausind)

######################################
#AMEND KINSHIP MATRIX TO MATCH PHENO
######################################

zz<-match(rownames(r.mat),pheno_ausind$Taxa);zz
which(is.na(zz))

#r.mat<-r.mat[-which(is.na(zz)),-which(is.na(zz))]
#dim(r.mat)

#check
match(unique(as.character(pheno_ausind$Taxa)),rownames(r.mat))-1:dim(r.mat)[1]



#export file
write.table(r.mat,'kinship_ausind.txt')


#######################
#STACKED PHENO FILE
#######################
head(pheno_ausind)
names(pheno_ausind)[2:25]
#create a file with control and salt stacked on each other
pheno_ausind_contr<-pheno_ausind[,c(1,2:5,10:13,18:21)]
dim(pheno_ausind_contr)#228  13
names(pheno_ausind_contr)
names(pheno_ausind_contr)<-c( "Taxa" ,       "RGR.2to9" , "RGR.2to6" , "RGR.6to9", 
 "RGR.9to13", "TR.2to9" ,  "TR.2to6" ,  "TR.6to9" , 
 "TR.9to13",  "TUE.2to9" , "TUE.2to6" , "TUE.6to9" ,
 "TUE.9to13")


pheno_ausind_salt<-pheno_ausind[,c(1,6:9,14:17,22:25)]
dim(pheno_ausind_salt)#228 13
names(pheno_ausind_salt)
names(pheno_ausind_salt)<-c(  "Taxa" ,   "RGR.2to9" , "RGR.2to6" , "RGR.6to9" ,
 "RGR.9to13", "TR.2to9" ,  "TR.2to6" ,  "TR.6to9" , 
 "TR.9to13" , "TUE.2to9" , "TUE.2to6" , "TUE.6to9" ,
 "TUE.9to13"
)

pheno_ausind_sc<-rbind(pheno_ausind_salt,pheno_ausind_contr)

#create salt/control treatment
pheno_ausind_sc$treat<-rep(c('s','c'),rep(dim(pheno_ausind_salt)[1],2))
pheno_ausind_sc$treat<-as.factor(pheno_ausind_sc$treat)



#########
#PCA DATA
#########

pc<-read.table('PC_INDAUS700.txt',header=T,skip=2)
head(pc)

pc$Taxa<-paste('X',pc$Taxa,sep='')

mm<-match(pc$Taxa,pheno_ausind$Taxa,)

ww<-which(is.na(mm));ww

pc<-pc[-ww,]
dim(pc)
dim(pheno_ausind)

#MERGE PHENO AND PC FILE

pheno_ausind_sc<-merge(pheno_ausind_sc,pc, by='Taxa')

#EXPORT AMMENDED PHENO FILE
write.csv(pheno_ausind_sc,'Master-File-Asreml-Nopanel_stacked_ausind.csv',row.names=F)



#############
#SNP DATA
#############
#note download tassel v5 and convert vcf to hapmap format

snp<-read.table('Hapmap_INDAUS700.maf005.nomissing.hmp.txt', header = T,as.is=T, comment.char = "")#,na.strings = "N")
dim(snp)
View(snp[1:20,1:20])

dim(snp)#5560 385
d1<-dim(snp)[1];d1
d2<-dim(snp)[2];d2
names(snp)[c(1,6)]<-c('rs' ,'assembly')

View(snp[1:5,1:20])
snp.mat<-snp[,12:(dim(snp)[2])]
#View(snp.mat)
rownames(snp.mat)<-snp$rs


nn<-match(names(snp.mat),pheno_ausind$Taxa)
which(!is.na(nn))
snp.mat<-snp.mat[,which(!is.na(nn))]
dim(snp.mat)
dim(r.mat)
dim(r.mat)*2
dim(pheno_ausind_sc)

write.table(snp.mat,'snp_ausind.txt')


#check number of allele options in snp
l<-apply(snp.mat,1,function(x) length(table(unlist(x))))
table(l)
#  2    3 
# 943 4617


#which heterozygotes
head(which(l==2))
head(which(l==3))
head(which(l==4))





write.csv(pc,'pc_ausind.csv')