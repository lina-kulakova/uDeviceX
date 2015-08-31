#!/bin/bash

#set -eu

# Usage:
#  ./aldriver2.sh <source directory> <parameter_line>

set -eu

source_directory=$1
parameter_line=$2

tmp_source_directory=`mktemp -d /tmp/alldriver.XXXX`
alpachio_config=`mktemp /tmp/alpachio.XXXX`
cart_list=`mktemp /tmp/cartlist.XXXX`

function msg() {
    printf "(aldriver.sh) %s\n" "$@"
}

function err() {
    msg "$@"
    exit 2
}

test -d "$source_directory" || \
    err "\"$source_directory\" is not a directory"

function run_case() {
    (cd "$tmp_source_directory"
     bash configs/daint/setup_daint.sh)
}

function create_case() {
    msg "create_case: $source_directory $tmp_source_directory"    
    test -d "$tmp_source_directory" && rm -rf "$tmp_source_directory"
    cp -r "$source_directory" "$tmp_source_directory"

    msg "config file: $alpachio_config"
    ./allineario.awk "$1" > "$alpachio_config"

    ./alpachio.sh "$alpachio_config" \
		  "$tmp_source_directory"/mpi-dpd/common.h \
		  "$tmp_source_directory"/cuda-rbc/rbc-cuda.cu \
		  "$tmp_source_directory"/configs/daint/setup_daint.sh \
		  "$tmp_source_directory"/configs/daint/vars.sh
    run_case
}

create_case "$parameter_line"
