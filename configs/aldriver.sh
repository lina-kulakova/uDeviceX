#!/bin/bash

set -eu

# Usage:
#   ./aldriver.sh <cartesian file> <source directory>

cart_file=$1
source_directory=$2
tmp_source_directory=`mktemp -d /tmp/alldriver.XXXX`
alpachio_config=`mktemp /tmp/alpachio.XXXX`

cart_list=/tmp/cart.list

function create_case() {
    echo "$source_directory" "$tmp_source_directory"    
    test -d "$tmp_source_directory" && rm -r "$tmp_source_directory"
    cp -r "$source_directory" "$tmp_source_directory"

    echo "$alpachio_config"
    ./allineario.awk $1 > "$alpachio_config"

    ./alpachio.sh "$alpachio_config" \
		  "$tmp_source_directory"/configs/setup_daint.sh \
		  "$tmp_source_directory"/mpi-dpd/common.h
}

./alcartesio.awk "${cart_file}" > "${cart_list}"
while read line
do
    create_case "${line}"
done < "${cart_list}"



