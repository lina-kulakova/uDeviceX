#!/bin/bash

default_dir=ctc-debug
script_name=configs/setup_daint.sh

rname=test1 #= rname=%my_dir_name% 

# remote host name
uname=lina
rhost="${uname}"@daint

# remote path name
rpath=/scratch/daint/"${uname}"/RBC/"${rname}"

err () {
    printf "(setup_daint.sh) $@\n"
    exit
}

test   -r "run_utils.sh" || err "cannot find run_utils.sh, I am in `pwd`"
.         "run_utils.sh"

# local top
ltop=$(git rev-parse --show-toplevel)

# current directory relative to the ltop
lcwd=$(git ls-files --full-name "${script_name}" | xargs dirname)

function move() {
    "${ltop}"/configs/gcp "${default_dir}" "${rhost}":"${rpath}"    
}

function compile() {
    rt "bash configs/compile_daint.sh"
}

function run() {
    rt "bash configs/run_daint.sh"
}

function post() {
    echo "mkdir -p ${rname} ; rsync -r -avz ${rhost}:${rpath}/${default_dir}/* ${rname}" >> ~/rsync.sh
    echo "${rname}"                                                                      >> ~/local.list
    echo "${rhost}:${rpath}/${default_dir}"                                              >> ~/remote.list    
}

move
compile
run
post
