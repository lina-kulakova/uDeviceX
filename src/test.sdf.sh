#### around cylinder
# cTEST: sdf.t1
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# cp sdf/cyl1/cyl.dat sdf.dat
# argp .conf.poiseuille.h \
#   -tend=4.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#   -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300 -pushtheflow > .conf.h
# make clean && make -j
# ./udx
# mid_h5.m h5/flowfields-0026.h5 | fhash.awk -v tol=2 > h5.out.txt
