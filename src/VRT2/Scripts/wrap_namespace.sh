# Tries to effect a namespace wrapping to avoid conflicts with functions
# in particular imaging libraries and the over-arching parameter estimation
# functions of Themis.
#
#!/bin/bash

# Get the user-supplied namespace name
NAMESPACE=$1

# Get the user-supplied file
FILE=$2

# Make sure we are all on the same page
echo "Wrapping $NAMESPACE around $FILE:---------------------"

TMPFILE1=wntmp1
if [ -f $TMPFILE1 ]; then
    rm $TMPFILE1
fi
touch $TMPFILE1

TMPFILE2=wntmp2
if [ -f $TMPFILE2 ]; then
    rm $TMPFILE2
fi
touch $TMPFILE2

typeset -i current_line=1

# If it is a .h file ...
if [ "${FILE:(-2)}" == ".h" ]; then

    # First look for the single define guard
    echo "  $FILE appears to be a header file"
    echo -n "    Looking for define guard:"

    guard_line_number=$(grep -n "#ifndef" $FILE | head -n1 | sed -e 's/:/ /' | gawk '{print $1}')

    if [ -z $guard_line_number ]; then
	echo " Couldn't find define guard."
	typeset -i guard_line_number=1
    else
	echo " Found define guard."
	typeset -i guard_line_number
	typeset -i current_line=$guard_line_number-1

	head -n$current_line $FILE >> $TMPFILE1

	old_guard_name=$(head -n$guard_line_number $FILE | tail -n1 | gawk '{print $2}')
	new_guard_name=$NAMESPACE'_'$old_guard_name
	
	echo "#ifndef $new_guard_name" >> $TMPFILE1
	echo "#define $new_guard_name" >> $TMPFILE1

	typeset -i current_line=$guard_line_number+2

	tail -n+$current_line $FILE >> $TMPFILE1
    fi

    # Second, look for first class definition
    echo -n "    Looking for class definitions:"
    first_definition_line_number=$(grep -n -e "class" -e "template" $TMPFILE1 | grep -v "\/\*" | head -n1 | sed -e 's/:/ /' | gawk '{print $1}')
    if [ -z $first_definition_line_number ]; then
	echo " Couldn't find any definitions to wrap."
	echo "  ERROR: Cowardly refusing to rename file."
	rm $TMPFILE1 $TMPFILE2
	typeset -i first_definition_line_number=1
    else
	echo " Found definitions to wrap."
	# Add namespace NAMESPACE {
	typeset -i first_definition_line_number
	typeset -i current_line=$first_definition_line_number-1
	head -n$current_line $TMPFILE1 >> $TMPFILE2
	echo "namespace $NAMESPACE {" >> $TMPFILE2
	typeset -i current_line=$first_definition_line_number
	tail -n+$current_line $TMPFILE1 >> $TMPFILE2

	mv $TMPFILE2 $TMPFILE1

	# Add };
	typeset -i endif_line=$(grep -n "#endif" $TMPFILE1 | tail -n1 | sed -e 's/:/ /' | gawk '{print $1}')
	typeset -i endif_line=endif_line-1
	head -n$endif_line $TMPFILE1 >> $TMPFILE2
	echo "};" >> $TMPFILE2
	typeset -i endif_line=endif_line+1
	tail -n+$endif_line $TMPFILE1 >> $TMPFILE2

        # Now take care of renaming
	echo "  Original file is in $FILE.unwrapped"
	cp -a $FILE $FILE.unwrapped
	mv $TMPFILE2 $FILE
	rm $TMPFILE1
    fi
elif [ "${FILE:(-4)}" == ".cpp" ]; then

    # First look for the single define guard
    echo "  $FILE appears to be a source file"
    echo -n "    Looking for function definitions:"
    
    first_definition_line_number=$(grep -n "::" $FILE | head -n1 | sed -e 's/:/ /' | gawk '{print $1}')

    if [ -z $first_definition_line_number ]; then
	echo " Couldn't find any function definitions."
	echo "  ERROR: Cowardly refusing to rename file."
	rm $TMPFILE1 $TMPFILE2
	typeset -i first_definition_line_number=1
    else
	echo " Found function definitions."
	typeset -i funcdef_line_number
	typeset -i current_line=$first_definition_line_number-1

	head -n$current_line $FILE >> $TMPFILE2
	echo "namespace $NAMESPACE {" >> $TMPFILE2
	typeset -i current_line=$first_definition_line_number
	tail -n+$current_line $FILE >> $TMPFILE2
	echo "};" >> $TMPFILE2

        # Now take care of renaming
	echo "  Original file is in $FILE.unwrapped"
	cp -a $FILE $FILE.unwrapped
	mv $TMPFILE2 $FILE
	rm $TMPFILE1
    fi    

else
    echo "  Type of $FILE is unrecognized, doing nothing."
fi
