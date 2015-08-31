#!/bin/bash

set -eu

# run this script from `configs' directory
# run all jobs in cart.dl.daint
gitroot=${HOME}/ctc-stretching

./alcartesio.awk cart.daint | \
    sh -x ./aldriver.sh "$gitroot"
