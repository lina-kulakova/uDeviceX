#!/usr/bin/awk -f

# [curr, icurr]
# [prev, iprev]

BEGIN {
    curr[1] = ""
    icurr = 1

    FIELD_SEP = "_"
}

function is_parameter() {
    return substr($0, 1, 1)=="="
}

function norm_name(s) {
    sub("^=", "", s)
    return s
}

function copy(src, dest, i) {
    for (i in src)
	dest[i] = src[i]
}

{ # strip comments
    ncomm = split($0, comm_arr, "#")
    if (ncomm>1)
	$0 = comm_arr[1]
}

!NF {
    next
}

is_parameter() {
    name = norm_name($1)

    copy(curr, prev)
    iprev = icurr
    icurr = 0
    delete curr
    
    next
}

{
	# create directory name
    val = $1
    for (i = 1; i<=iprev; i++) {
	sep = prev[i] ? FIELD_SEP : ""
	curr[++icurr] = prev[i] sep name FIELD_SEP val
    }
}

END {
    for (i = 1; i<=icurr; i++)
	print curr[i]
}
