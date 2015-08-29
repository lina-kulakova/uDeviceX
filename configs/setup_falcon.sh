#!/bin/bash

cp configs/Makefile.falcon mpi-dpd/.cache.Makefile

opath=$PATH

PATH=/usr/local/cuda/bin/:$PATH

cd mpi-dpd
make clean && make -j slevel="-2"
