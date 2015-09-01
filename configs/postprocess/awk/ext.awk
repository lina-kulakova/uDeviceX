#!/usr/bin/awk -f

# Input
# argument: directory name
#
# Outputs
# `istep' `xext' `yext' `zext'

BEGIN {
    dir=ARGV[1]

    # get the last .o file
    cmd = sprintf("ls -1t %s/rbc_stretching.*.o | head -n1", dir)
    cmd | getline f

    ARGV[1] = f
}

$1=="RBC" && $2=="diameters:" {
    xext = $3; yext=$4; zext=$5
    print ++istep, xext, yext, zext
}
