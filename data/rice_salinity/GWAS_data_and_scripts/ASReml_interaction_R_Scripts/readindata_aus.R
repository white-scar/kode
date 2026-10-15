#FILE THAT AMENDS AND EXPORTS DATA SO READY FOR IMPORT 


#####################
#KINSHIP MATRIX
#####################

r.df<-read.table('Kin_AUS700.maf005.nomissing.txt',header=F,skip=1)


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
solve(r.mat)
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

pheno_aus<-pheno

#############################################################
#AMEND TRAIT FILE (pheno_aus) TO JUST CONTAIN WHATS IN THE KINSHIP MATRIX
#############################################################

mm<-match(rownames(r.mat),unique(as.character(pheno_aus$Taxa)))
mm[which(!is.na(mm))]
pheno_aus<-pheno_aus[mm[which(!is.na(mm))],]
dim(pheno_aus)

######################################
#AMEND KINSHIP MATRIX TO MATCH PHENO
######################################

zz<-match(rownames(r.mat),pheno_aus$Taxa);zz
which(is.na(zz))

#r.mat<-r.mat[-which(is.na(zz)),-which(is.na(zz))]
#dim(r.mat)

#check
match(unique(as.character(pheno_aus$Taxa)),rownames(r.mat))-1:dim(r.mat)[1]



#export file
write.table(r.mat,'kinship_aus.txt')


#######################
#STACKED PHENO FILE
#######################
head(pheno_aus)
names(pheno_aus)[2:25]
#create a file with control and salt stacked on each other
pheno_aus_contr<-pheno_aus[,c(1,2:5,10:13,18:21)]
dim(pheno_aus_contr)#228  13
names(pheno_aus_contr)
names(pheno_aus_contr)<-c( "Taxa" ,       "RGR.2to9" , "RGR.2to6" , "RGR.6to9", 
 "RGR.9to13", "TR.2to9" ,  "TR.2to6" ,  "TR.6to9" , 
 "TR.9to13",  "TUE.2to9" , "TUE.2to6" , "TUE.6to9" ,
 "TUE.9to13")


pheno_aus_salt<-pheno_aus[,c(1,6:9,14:17,22:25)]
dim(pheno_aus_salt)#228 13
names(pheno_aus_salt)
names(pheno_aus_salt)<-c(  "Taxa" ,   "RGR.2to9" , "RGR.2to6" , "RGR.6to9" ,
 "RGR.9to13", "TR.2to9" ,  "TR.2to6" ,  "TR.6to9" , 
 "TR.9to13" , "TUE.2to9" , "TUE.2to6" , "TUE.6to9" ,
 "TUE.9to13"
)

pheno_aus_sc<-rbind(pheno_aus_salt,pheno_aus_contr)

#create salt/control treatment
pheno_aus_sc$treat<-rep(c('s','c'),rep(dim(pheno_aus_salt)[1],2))
pheno_aus_sc$treat<-as.factor(pheno_aus_sc$treat)




#########
#PCA DATA
#########

pc<-read.table('PC_AUS700.txt',header=T,skip=2)
head(pc)

pc$Taxa<-paste('X',pc$Taxa,sep='')

mm<-match(pc$Taxa,pheno_ausind$Taxa,)

ww<-which(is.na(mm));ww

#pc<-pc[-ww,]
dim(pc)
dim(pheno_aus)


#MERGE PHENO AND PC FILE

pheno_aus_sc<-merge(pheno_aus_sc,pc, by='Taxa')





#EXPORT AMMENDED FILE
write.csv(pheno_aus_sc,'Master-File-Asreml-Nopanel_stacked_aus.csv',row.names=F)




#############
#SNP DATA
#############
#note download tassel v5 and convert vcf to hapmap format

snp<-read.table('Hapmap_AUS700.maf005.nomissing.hmp.txt', header = T,as.is=T, comment.char = "")#,na.strings = "N")
dim(snp)
#View(snp)

dim(snp)#16538 157
d1<-dim(snp)[1];d1
d2<-dim(snp)[2];d2
names(snp)[c(1,6)]<-c('rs' ,'assembly')

View(snp[1:5,1:20])
snp.mat<-snp[,12:(dim(snp)[2])]
#View(snp.mat)
rownames(snp.mat)<-snp$rs


nn<-match(names(snp.mat),pheno_aus$Taxa)
which(!is.na(nn))
snp.mat<-snp.mat[,which(!is.na(nn))]
dim(snp.mat)
dim(r.mat)
dim(r.mat)*2
dim(pheno_aus_sc)

write.table(snp.mat,'snp_aus.txt')


#check number of allele options in snp
l<-apply(snp.mat,1,function(x) length(table(unlist(x))))
table(l)
#   2     3 
# 6296 10242

#previous
#  2     3     4 
#  842 22527 23239 

#which heterozygotes
head(which(l==2))
head(which(l==3))
head(which(l==4))


