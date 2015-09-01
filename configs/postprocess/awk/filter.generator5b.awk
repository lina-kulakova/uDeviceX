#!/usr/bin/awk -f

BEGIN {
    file = ARGV[1]
    ARGV[1] = ""

    printf "    if (__rc)\n"
    printf "        print \"%s\"\n", file
    printf "}\n"

}

