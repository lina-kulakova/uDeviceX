#!/usr/bin/awk -f

# Input
# `istep' `xext' `yext' `zext'

# Outputs
# average extensions in the following format
#  xext_av 1.0
#  yext_av 2.0

BEGIN {
    d = length(d) ? d : 10
}

{
    xext = $2; yext=$3; zext=$4
    
    xext_sum += xext; yext_sum += yext; zext_sum += zext
    n++
}


function print_par(key, val) {
    print key, val
}

END {
    print_par("xext_av", xext_sum/n)
    print_par("yext_av", yext_sum/n)
    print_par("zext_av", zext_sum/n)    
}

