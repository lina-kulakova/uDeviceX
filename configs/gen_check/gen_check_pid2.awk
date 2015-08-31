#!/usr/bin/awk -f

# adds two columns to the input 'count_from_start' and count_from_end' (1...N)

{
    arr[++i] = $0
}

END {
    for (k=1; k<=i; k++)
	print k, i-k+1, arr[k]
}
