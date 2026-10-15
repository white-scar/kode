#! /bin/bash
#pwd
ls *.csv > Trait_Names

mkdir ../Results


mkdir -v ../Results/CHR

for file in $(cat ./Trait_Names)
do
prefix=${file/.txt/}
echo "processing file : $file"
sed -e 's/	/,/g' $file > Input.csv 

r BATCH -f ../Chr_bp.r 

mv CHR.csv ../Results/CHR/$prefix.csv


echo " done ! $prefix"
done