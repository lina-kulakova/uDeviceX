#!/bin/bash

#set -eu

# Usage:
#   ./aldriver.sh <source directory> <cartesian file>

source_directory=$1

cart_file=/dev/stdin
if test $# -eq 2; then
    cart_file=$2
fi

xargs -n 1 ./aldriver2.sh "$source_directory" < "$cart_file"
