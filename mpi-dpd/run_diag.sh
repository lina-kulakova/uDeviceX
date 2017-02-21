#!/bin/bash

# Run from this directory:
#  > atest run_diag.sh
#
# To update the test change TEST to cTEST and run
#  > atest run_diag.sh
# add crap from test_data/* to git
#
# cTEST: diag.t1
# set -x
# export PATH=../tools:$PATH
# cp .conf.test.h .conf.h
# echo 0 0 0  1 0 0 8  0 1 0 8  0 0 1 8  0 0 0 1 > rbcs-ic.txt
# make clean && make -j && make -C ../tools
# rm -rf ply h5 diag.txt
# ./test 1 1 1 -rbcs -tend=0.5 -steps_per_dump=100
# awk '{print 100*$2}' diag.txt    | fhash.awk -v tol=2 > diag.out.txt
#
# cTEST: diag.t2
# set -x
# export PATH=../tools:$PATH
# cp .conf.test.h .conf.h
# echo 0 0 0  1 0 0 8  0 1 0 8  0 0 1 8  0 0 0 1 > rbcs-ic.txt
# make clean && make -j && make -C ../tools
# rm -rf ply h5 diag.txt
# ./test 1 1 1 -rbcs -tend=0.5 -steps_per_dump=100
# ply2punto ply/rbcs-00009.ply | fhash.awk -v tol=2 > ply.out.txt
#
# cTEST: diag.t3
# export PATH=../tools:$PATH
# cp .conf.test.h .conf.h
# cp sdf/wall1/wall.dat                               sdf.dat
# x=0.75 y=8 z=12
# echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# make clean && make -j && make -C ../tools
# rm -rf ply h5 diag.txt
# ./test 1 1 1 -rbcs -tend=0.5 -steps_per_dump=300  -walls  -wall_creation_stepid=100 \
#       -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300
# ply2punto ply/rbcs-00003.ply | fhash.awk -v tol=1 > ply.out.txt
#
# cTEST: diag.t4
# export PATH=../tools:$PATH
# cp .conf.test.h .conf.h
# cp sdf/wall1/wall.dat sdf.dat
# make clean && make -j && make -C ../tools
# rm -rf ply h5 diag.txt
# ./test 1 1 1 -tend=2.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#       -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300
# avg_h5.m h5/flowfields-0006.h5 | fhash.awk -v tol=1 > h5.out.txt
#
# cTEST: diag.t5
# export PATH=../tools:$PATH
# cp .conf.poiseuille.h .conf.h
# cp sdf/wall1/wall.dat sdf.dat
# make clean && make -j && make -C ../tools
# rm -rf ply h5 diag.txt
# ./test 1 1 1 -tend=2.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#       -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300 -pushtheflow
# avg_h5.m h5/flowfields-0013.h5 | fhash.awk -v tol=2 > h5.out.txt
#
# cTEST: diag.t6
# export PATH=../tools:$PATH
# cp .conf.poiseuille.h .conf.h
# cp sdf/cyl1/cyl.dat sdf.dat
# make clean && make -j && make -C ../tools
# rm -rf ply h5 diag.txt
# ./test 1 1 1 -tend=4.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#       -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300 -pushtheflow
# mid_h5.m h5/flowfields-0026.h5 | fhash.awk -v tol=2 > h5.out.txt
#
# cTEST: flow.around.t1
# export PATH=../tools:$PATH
# cp .conf.around.h  .conf.h
# cp sdf/cyl1/cyl.dat sdf.dat
# x=0.75 y=8 z=9
# echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 > rbcs-ic.txt
# make clean && make -j && make -C ../tools
# rm -rf ply h5 diag.txt
# ./test 1 1 1 -rbcs -tend=4.0 -steps_per_dump=5000 -walls -wall_creation_stepid=1000 \
#       -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=5000 -pushtheflow
# mid_h5.m h5/flowfields-0001.h5 | fhash.awk -v tol=1 > h5.out.txt
#
# a test case with two RBCs around cylinder
# cTEST: flow.around.t2
# export PATH=../tools:$PATH
# cp .conf.around.h  .conf.h
# cp sdf/cyl1/cyl.dat sdf.dat
# x=0.75  y=3 z=9; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >  rbcs-ic.txt
# x=0.75 y=13 z=9; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >> rbcs-ic.txt
# make clean && make -j && make -C ../tools
# rm -rf ply h5 diag.txt
# ./test 1 1 1 -rbcs -tend=4.0 -steps_per_dump=5000 -walls -wall_creation_stepid=1000 \
#        -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=5000 -pushtheflow
# ply2punto ply/rbcs-00001.ply | fhash.awk -v tol=1 > ply.out.txt
#
# a test case with two RBCs around cylinder with one RBC removed by the wall
# cTEST: flow.around.t3
# export PATH=../tools:$PATH
# cp .conf.around.h  .conf.h
# cp sdf/cyl1/cyl.dat sdf.dat
# x=0.75 y=3 z=9; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >  rbcs-ic.txt
# x=8    y=8 z=8; echo 0 0 0  1 0 0 $x  0 1 0 $y  0 0 1 $z  0 0 0 1 >> rbcs-ic.txt
# make clean && make -j && make -C ../tools
# rm -rf ply h5 diag.txt
# ./test 1 1 1 -rbcs -tend=4.0 -steps_per_dump=5000 -walls -wall_creation_stepid=1000 \
#        -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=5000 -pushtheflow
# ply2punto ply/rbcs-00001.ply | fhash.awk -v tol=1 > ply.out.txt
