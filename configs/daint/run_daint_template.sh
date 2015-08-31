#!/bin/bash -l
#
#SBATCH --job-name="dpd_kolmogorov"
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=dpd_kolmogorov.%j.o
#SBATCH --error=dpd_kolmogorov.%j.e

#======START=====
args=$1

module swap PrgEnv-cray PrgEnv-gnu
module load cudatoolkit
module load cray-hdf5-parallel
module load cray-mpich

aprun ./test ${args}
#=====END====
