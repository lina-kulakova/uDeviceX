#!/usr/bin/awk -f

$1 == "addstf:" {
    pid=$2
    sgn=1
    arr[pid]+=sgn
}

END {
    for (pid in arr)
	if (arr[pid]!=0)
	    print pid, arr[pid]
}
