#!/bin/bash -l
#
#SBATCH --job-name="rbc_stretching"
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=rbc_stretching.%j.o
#SBATCH --error=rbc_stretching.%j.e

#======START=====
source configs/daint/vars.sh
source configs/daint/load_modules.sh

cd mpi-dpd
aprun ./test ${args}
#=====END====
