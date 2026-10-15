#! /bin/bash
#pwd
ls *.csv > Trait_Names

mkdir ../Results

mkdir -v ../Results/Plot

for file in $(cat ./Trait_Names)
do
prefix=${file/.txt/}
echo "processing file : $file"
sed -e 's/	/,/g' $file > Input.csv 

r BATCH -f ../MANPLOTAsreml.r 



mv markplot.BON.jpeg ../Results/Plot/$prefix.markplot.BON.jpeg
mv marktrtplot.BON.jpeg ../Results/Plot/$prefix.marktrtplot.BON.jpeg

echo " done ! $prefix"
done