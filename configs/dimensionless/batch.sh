#!/bin/bash

set -eu

# run this script from `configs' directory
# run all jobs in cart.dl.daint
gitroot=/home/lisergey/work/kulina/kolmogorov

./alcartesio.awk dimensionless/cart.dl.daint | \
    ./altransformio-pipe.sh dimensionless/cart.transform | \
    sh -x ./aldriver-pipe.sh "$gitroot"
