#!/bin/bash


TARGETDIR="../include/vrt2"
SUBDIRS=$(ls -d */)

# Make include directory
mkdir -p $TARGETDIR

# Copy all of the project headers to include/vrt2
for dir in $SUBDIRS; do
    cp -a $dir*.h $TARGETDIR
done

# Go through the project headers and append a vrt2/ to them
cd $TARGETDIR
HFILES=$(ls *.h)
ls *.h > tmp1
for HFA in $HFILES; do

    grep "^\#include .*\.h" $HFA | awk '{print $2}' | sed -e 's/"//g' > tmp2
    VHF=$(awk 'NR==FNR{arr[$0];next} $0 in arr' tmp1 tmp2)

    for HFB in $VHF; do
	cat $HFA | sed -e "s/$HFB/vrt2\/$HFB/" > tmp
	mv tmp $HFA
    done
    echo Copied $HFA to ../include/vrt2
done

rm tmp1 tmp2

cd ..
echo "/// VRT2 header file" > vrt2.h
echo "/// Programs must include vrt2.h and link to libvrt2.a" >> vrt2.h
echo "" >> vrt2.h
echo "#ifndef VRT2_H" >> vrt2.h
echo "#define VRT2_H" >> vrt2.h
echo "" >> vrt2.h
ls */*.h | awk '{printf "#include \"%s\"\n",$1}' >> vrt2.h
echo "" >> vrt2.h
echo "#endif" >> vrt2.h
