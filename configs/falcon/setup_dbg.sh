#!/bin/bash

. configs/falcon/vars.sh

PATH=/usr/local/cuda/bin/:$PATH

function err() {
    printf "(setup_falcon.sh) $@\n"
    exit
}

flist="./mpi-dpd/Makefile"


function ctrl_c() {
        configs/restore.sh $flist
}

function compile() {
    configs/backup.sh  $flist
    configs/replace.sh '-O[234]'  '-O0' $flist
    configs/replace.sh '-DNDEBUG' ''    $flist

    # trap C-c so I can restore Makefile
    trap ctrl_c INT
    configs/falcon/compile.sh
    trap -      INT

    configs/restore.sh $flist
    mv mpi-dpd/test mpi-dpd/test_dbg
}

compile
configs/falcon/postcompile.sh
configs/falcon/preproc.sh
configs/falcon/run_dbg.sh
