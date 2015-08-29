#!/bin/bash

# al-patch files in-place
#   Usage:
#   ./alpachio.sh <config_file> [file1 file2 file3]
#
# TEST: alpachio1
# cp test_data/common.cpp common.tmp.cpp
# ./alpachio.sh test_data/alpatchi1.config common.tmp.cpp
# cp common.tmp.cpp       common.out.cpp
#
# TEST: alpachio2
# cp test_data/common2.cpp common.tmp.cpp
# ./alpachio.sh test_data/alpatchi1.config common.tmp.cpp
# cp common.tmp.cpp       common.out.cpp


config=$1
shift

t=`mktemp /tmp/al.XXXXX`
for f ; do
    ./alpachio.awk "$config" "$f" >  "$t"
    mv             "$t"              "$f"
done
