#!/bin/bash

. configs/falcon/vars.sh
cd mpi-dpd
#./test $args
sbatch run_daint_template.sh $args
