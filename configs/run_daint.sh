#!/bin/bash -l

mv mpi-dpd/test .
sbatch configs/run_daint_template.sh
