#!/bin/bash

# update tags
#

#PATH=$PATH:$HOME/prefix-pkgsrc-2015Q1/bin

dlist="cuda-ctc cuda-dpd-sem cuda-rbc logistic_rng mpi-dpd"
exctags --language-force=c++ \
	-e `find  ${dlist} '(' -name '*.h' -or -name '*.cpp' -or -name '*.cu' ')' `
