#!/bin/bash

default_dir=ctc-debug
script_name=gcp

rname=test1

# remote host name
uname=lina
rhost="${uname}"@daint

# remote path name
rpath=/scratch/daint/"${uname}"/RBC/"${rname}"

source "run_utils.sh"

# local top
ltop=$(git rev-parse --show-toplevel)

# current directory relative to the ltop
lcwd=$(git ls-files --full-name "${script_name}" | xargs dirname)

function move() {
    "${ltop}"/configs/gcp "${default_dir}" "${rhost}":"${rpath}"    
}


move


