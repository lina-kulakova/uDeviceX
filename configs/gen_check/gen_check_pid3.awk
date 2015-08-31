#!/usr/bin/awk -f

# assign a sign of the force to atoms
#  Ouputs
# `id' `force'

BEGIN {
    N = length(N) ? N : 10
}

{
    from_start = $1
    from_end   = $2
    x          = $3
    id         = $4

    if (from_start <= N) {
	force = -1
	print id, force
    } else if (from_end <= N) {
	force = 1
	print id, force
    }
}
