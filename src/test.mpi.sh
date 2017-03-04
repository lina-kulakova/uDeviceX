#!/bin/bash

#### Periodic box (mpi on falcon)
# TEST: mpi.t1
# set -x
# export PATH=../tools:$PATH
# export PATH=/usr/lib64/mpich/bin:$PATH
# rm -rf ply h5 diag.txt
# argp .conf.test.h -tend=0.5 -steps_per_dump=100 > .conf.h
# make clean && make -j && make -C ../tools
# mpirun -n 2 ./udx 2 1 1
# awk '{print 100*$2}' diag.txt    | fhash.awk -v tol=2 > diag.out.txt

#### Double poiseuille
# TEST: mpi.t2
# set -x
# export PATH=../tools:$PATH
# export PATH=/usr/lib64/mpich/bin:$PATH
# rm -rf ply h5 diag.txt
# argp .conf.double.poiseuille.h     \
#   -tend=2.0 -steps_per_dump=100    \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -hdf5part_dumps \
#   -steps_per_hdf5dump=300 -pushtheflow > .conf.h
# make clean && make -j && make -C ../tools
# mpirun -n 2 ./udx 2 1 1
# avg_h52.m h5/flowfields-0013.h5 | fround.awk -v tol=2 > h5.out.txt

####
# TEST: mpi.t3
# export PATH=../tools:$PATH
# export PATH=/usr/lib64/mpich/bin:$PATH
# rm -rf ply h5 diag.txt
# cp sdf/cyl1/cyl.dat sdf.dat
# argp .conf.around.h \
#    -tend=4.0 -steps_per_dump=5000 -walls -wall_creation_stepid=1000 \
#    -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=5000 -pushtheflow > .conf.h
# make clean && make -j && make -C ../tools
# mpirun -n 3 ./udx 1 1 3
# mid_h5.m h5/flowfields-0001.h5 | fhash.awk -v tol=1 > h5.out.txt
#
