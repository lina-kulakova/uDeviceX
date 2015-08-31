#!/bin/bash

. configs/daint/vars.sh

#opath=$PATH
#PATH=/usr/local/cuda/bin/:$PATH

function err() {
    printf "(setup_daint.sh) $@\n"
    exit
}

configs/daint/compile.sh
configs/daint/preproc.sh
configs/daint/run.sh
