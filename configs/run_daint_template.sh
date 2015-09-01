#!/bin/bash -l
#
#SBATCH --job-name="dpd_kolmogorov"
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=dpd_kolmogorov.%j.o
#SBATCH --error=dpd_kolmogorov.%j.e

#======START=====
module swap PrgEnv-cray PrgEnv-gnu
module load cudatoolkit
module load cray-hdf5-parallel
module load cray-mpich

aprun ./test 1 1 1
#=====END====
