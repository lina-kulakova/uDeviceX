#!/bin/bash

. configs/daint/vars.sh

PATH=/usr/local/cuda/bin/:$PATH

function err() {
    printf "(setup_daint.sh) $@\n"
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
    configs/daint/compile.sh
    trap -      INT

    configs/restore.sh $flist
    mv mpi-dpd/test mpi-dpd/test_dbg
}

compile
configs/daint/postcompile.sh
configs/daint/preproc.sh
configs/daint/run_dbg.sh
