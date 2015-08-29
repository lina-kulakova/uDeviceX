#!/bin/bash

opath=$PATH
PATH=/usr/local/cuda/bin/:$PATH

function err() {
    printf "(setup_falcon.sh) $@\n"
    exit
}

function compile() {
    cp configs/Makefile.falcon mpi-dpd/.cache.Makefile
    cp configs/Makefile.falcon cuda-ctc/.cache.Makefile
    cd mpi-dpd
    make -j clean && make slevel="-2"
    cd -
}

function run() {
    ./mpi-dpd/test 1 1 1 -rbc -stretching_force 10
}

compile
run
