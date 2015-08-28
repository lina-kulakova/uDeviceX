#!/bin/bash

cp configs/Makefile.daint mpi-dpd/.cache.Makefile

module swap PrgEnv-cray PrgEnv-gnu
module load cudatoolkit
module load cray-hdf5-parallel
module load cray-mpich

cd mpi-dpd
make clean && make -j slevel="-2"
cd -
