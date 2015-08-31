#!/bin/bash

# This file updates ../../cuda-rbc/check_pid_stretching_body.h
# we pull `N' atoms on one side an `N' atoms on the other
# Usage:
# ./gen_check.sh 10

N=$1
rbc=../../cuda-rbc/rbc.dat

./gen_check_pid1.awk $rbc        | \
    ./gen_check_pid2.awk         | \
    ./gen_check_pid3.awk -v N=$N | \
    ./gen_check_pid4.awk $rbc    | \
    ./gen_check_pid5.awk -v N=$N | \
    ./gen_check_pid6.awk         | \
    tee  ../../cuda-rbc/check_pid_stretching_body.h

