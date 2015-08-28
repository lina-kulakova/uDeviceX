#!/bin/bash

#set -eu

# Usage:
#   ./aldriver.sh <cartesian file> <source directory>

cart_file=$1
source_directory=$2
tmp_source_directory=`mktemp -d /tmp/alldriver.XXXX`
alpachio_config=`mktemp /tmp/alpachio.XXXX`

cart_list=`mktemp /tmp/cartlist.XXXX`

function msg() {
    printf "(aldriver.sh) %s\n" "$@"
}

function run_case() {
    cd "$tmp_source_directory"
    bash configs/setup_daint.sh
    cd -
}

function create_case() {
    msg "create_case: $source_directory $tmp_source_directory"    
    test -d "$tmp_source_directory" && rm -rf "$tmp_source_directory"
    cp -r "$source_directory" "$tmp_source_directory"

    msg "config file: $alpachio_config"
    ./allineario.awk $1 > "$alpachio_config"

    ./alpachio.sh "$alpachio_config" \
		  "$tmp_source_directory"/configs/setup_daint.sh \
		  "$tmp_source_directory"/mpi-dpd/common.h
    run_case
}

./alcartesio.awk "${cart_file}" > "${cart_list}"

for line in `cat "${cart_list}"`; do
    create_case "${line}"
done
