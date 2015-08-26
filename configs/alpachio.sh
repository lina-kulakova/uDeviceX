#!/bin/bash

# al-patch files in-place
#   Usage:
#   ./alpachio.sh <confg> [file1 file2 file3]
#


config=$1
shift

t=`mktemp /tmp/al.XXX`
for f ; do
    cp              "$f"             "$f".bak
    ./alpachio.awk "$config" "$f" >  "$t"
    mv             "$t"              "$f"
done
