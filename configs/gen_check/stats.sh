#!/bin/bash

# prints the number of bonds, number of angles, and number of
# dihedrals for each atom

rbc=../../cuda-rbc/rbc.dat
natoms=`awk 'NR==1{print $1}' "$rbc"`
seq 1  $natoms | ./gen_check_pid4.awk "$rbc"
    

