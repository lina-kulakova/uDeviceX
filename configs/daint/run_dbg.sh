#!/bin/bash

. configs/daint/vars.sh
cd mpi-dpd

printf "(run_dbg.sh) to run gdb\n"
echo  "gdb ./test_dbg"
echo  "r $args"
#./test_dbg
