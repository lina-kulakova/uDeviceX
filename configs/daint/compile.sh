#!/bin/bash

cp configs/daint/Makefile mpi-dpd/.cache.Makefile
cp configs/daint/Makefile cuda-ctc/.cache.Makefile
cd mpi-dpd
source ../configs/daint/load_modules.sh
make clean && make -j slevel="-2"
cd -
