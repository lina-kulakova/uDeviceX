#!/bin/bash

. configs/falcon/vars.sh

(
    cd cell-placement
    make
    ./cell-placement $XSIZE_SUBDOMAIN $YSIZE_SUBDOMAIN $ZSIZE_SUBDOMAIN $xranks $yranks $zranks
)

cp ./cell-placement/rbcs-ic.txt mpi-dpd/
