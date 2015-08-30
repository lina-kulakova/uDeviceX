#!/bin/bash

. configs/falcon/vars.sh

opath=$PATH
PATH=/usr/local/cuda/bin/:$PATH

function err() {
    printf "(setup_falcon.sh) $@\n"
    exit
}

configs/falcon/compile.sh
configs/falcon/preproc.sh
configs/falcon/run.sh
