#!/usr/bin/awk -f

# Input
# `istep' `xext' `yext' `zext'
# argument `nd': the total number of timestps
# argument ` d': the number timesteps to keep at the end

# Outputs
# last

BEGIN {
    d = length(d) ? d : 10
}

{
    istep=$1; xext = $2; yext=$3; zext=$4
    from_end = nd - istep + 1
    if (from_end <= d)
	print istep, xext, yext, zext
}

