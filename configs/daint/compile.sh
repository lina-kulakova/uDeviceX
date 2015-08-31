#!/bin/bash

cp configs/falcon/Makefile mpi-dpd/.cache.Makefile
cp configs/falcon/Makefile cuda-ctc/.cache.Makefile
cd mpi-dpd
make -j clean && make slevel="-2"
