#! /bin/bash
#pwd
ls *.csv > Trait_Names

mkdir ../Results
mkdir -v ../Results/FDR
mkdir -v ../Results/BON
mkdir -v ../Results/Plot

for file in $(cat ./Trait_Names)
do
prefix=${file/.txt/}
echo "processing file : $file"
sed -e 's/	/,/g' $file > Input.csv 

r BATCH -f ../MANPLOTAsreml.r 

mv FDR.csv ../Results/FDR/$prefix.FDR.csv
mv markplot.BON.csv ../Results/BON/$prefix.markplot.BON.csv
mv marktrtplot.BON.csv ../Results/BON/$prefix.marktrtplot.BON.csv

mv markplot.BON.jpeg ../Results/Plot/$prefix.markplot.BON.jpeg
mv marktrtplot.BON.jpeg ../Results/Plot/$prefix.marktrtplot.BON.jpeg

echo " done ! $prefix"
done