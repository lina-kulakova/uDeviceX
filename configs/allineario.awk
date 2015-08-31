#!/usr/bin/awk -f

# Make alpachio.awk config file from parameters string
#
# TEST: allineario1
# ./allineario.awk  a_10_b_20 > allineario.out.config
#
# TEST: allineario2
# ./allineario.awk  a_10_aa_preved > allineario.out.config

BEGIN {
    FIELD_SEP = "_"
    
    nn = split(ARGV[1], arr, FIELD_SEP)

    for (i = 1; i<=nn; i+=2) {
	key=arr[i]
	val=arr[i+1]
	printf "%s %s\n", key, val
    }
}
