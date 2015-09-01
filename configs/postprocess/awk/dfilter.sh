#!/bin/bash

filter_file=$1
shift

t=`mktemp /tmp/dfilter.XXXX`

for f
do
    b=`basename $f`
    
    
    ../../allineario.awk  "$b" > "$t"
    r=`./filter1.sh "$filter_file" "$t"`

    test "$r" && echo "$f"

done
