#!/bin/bash

cp configs/Makefile.falcon mpi-dpd/.cache.Makefile
cp configs/Makefile.falcon cuda-ctc/.cache.Makefile

opath=$PATH

PATH=/usr/local/cuda/bin/:$PATH

cd mpi-dpd
make -j clean && make slevel="-2"
cd -

./mpi-dpd/test 1 1 1 -rbc -stretching_force 10
