#!/usr/bin/awk -f

# Make alpachio.awk config file from a line

BEGIN {
    FIELD_SEP = "_"
    
    nn = split(ARGV[1], arr, FIELD_SEP)

    for (i = 1; i<=nn; i+=2) {
	print arr[i], arr[i+1]
    }
}
