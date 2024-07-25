# Runs through all files in a directory and wraps them all

#!/bin/bash

NAMESPACE=$1
FILES=$(ls *.h *.cpp)

for FILE in $FILES; do
    /home/abroderi/Themis/Themis/src/vrt2_lib/src/Scripts/wrap_namespace.sh $NAMESPACE $FILE
done