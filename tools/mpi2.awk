#!/usr/bin/awk -f

BEGIN {
    fu = "MPI_[A-Za-z_]"
    sp = "[\n\t]*"
    br = "\\("

    pat = fun sp br
}

{
    sep = NR == 1 ? "" : ORS
    f = f sep $0 
}

END {
    #print f

    match(f, pat)
    print substr(f, RSTART, RLENGTH)
}
