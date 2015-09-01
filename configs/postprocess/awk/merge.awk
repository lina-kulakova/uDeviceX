#!/usr/bin/awk -f

# merge several parameter files
# Usage:
# ./merge.awk [file ...]

{
   sub(/#.*/, "")         # strip comments
}


NF >= 2 {
    key=$1
    val=$2

    if (key in arr) {
	if (arr[key]!=val) {
	    printf ("(merge.awk) I see %s = %s at %s:%s, but it was %s befor\n",
		    key, val, FILENAME, FNR, arr[key])
	    rc_code = 1
	    exit rc_code
	}
	next
    }

    arr[key]    = val
    idx[++ipar] = key
}

function print_par(key, val) {
    print key, val
}

END {
    if (rc_code) exit rc_code
    for (i=1; i<=ipar; i++) {
	key = idx[i]
	val = arr[key]
	print_par(key, val)
    }
}
