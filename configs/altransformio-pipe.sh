#!/bin/bash
transform_file=$1

function msg() {
    printf "(altransformio-pipe.sh) %s\n" "$@"
}

function err() {
    msg "$@"
    exit 2
}

infile=/dev/stdin
if test $# -gt 3; then
    infile=$2
fi

test -r "$transform_file" || err "\"$transform_file\" is not a file"
test -r "$infile"         || err "\"$infile\" is not a file"

xargs < "$infile" -n 1 ./altransformio.sh "$transform_file" 
