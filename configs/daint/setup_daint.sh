#!/bin/bash

default_dir=ctc-stretching
script_name=configs/setup_daint.sh

rname=test3 #= rname=%my_dir_name% 

# remote host name
uname=lina
rhost="${uname}"@daint

# remote path name
rpath=/scratch/daint/"${uname}"/RBCstretcing/"${rname}"

function err () {
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

function setup() {
	rt "configs/daint/setup.sh"
}

function post() {
    echo "mkdir -p ${rname} ; rsync -r -avz ${rhost}:${rpath}/${default_dir}/* ${rname}" >> ~/stretching_rsync.sh
    echo "${rname}"                                                                      >> ~/stretching_local.list
    echo "${rhost}:${rpath}/${default_dir}"                                              >> ~/stretching_remote.list    
}

move
setup
post
