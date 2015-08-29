#!/bin/bash

# do batch replace
# Usage:
# ./replace.sh A B [f1 ...]

old=$1; shift
new=$1; shift

t=`mktemp /tmp/rep.XXX`

for f ; do
    sed  -e "s,$old,$new,g" "$f" >  "$t"
    mv "$t" "$f"
done
