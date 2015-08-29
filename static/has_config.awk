#!/usr/bin/awk -f
#
# Check if #include "config.h" and defs are used consistently
# Usage:
# ./has_config.awk ../mpi-dpd/*.cu  ../mpi-dpd/*.h ../mpi-dpd/*.cu
# ./has_config.awk `find .. '(' -name '*.h' -or -name '*.cpp' -or -name '*.h' ')'`

BEGIN {
    pat = "#include\"config.h\""
    # alist of variables to check for
    # TODO:
    defs_str  = "DO_STRETCHING"
    asplit(defs_str, defs)
}

function asplit(str, arr) {  # make an assoc array from str
    n = split(str, temp)
    for (i = 1; i <= n; i++)
        arr[temp[i]]++
    return n
}

function is_def_use(i, safe_zero) {
    # remove strings
    gsub(/"([^"]|\\")*"/, "")
    if (substr($1, 1, 1) != "#")
	return 0

    for (i=2; i<=NF; i++)
	if ($i in defs) return 1
    
    return 0
}

function is_include() {
    return $1 == "#include" && $2 == "\"config.h\""
}

function end_file() {
    if (last_use && !last_include)
	printf "%s:%d: error: has definition but does not include config.h\n", last_file, last_use
    
    if (!last_use && last_include)
	printf "%s:%d: error: includes config.h but does not have any definitions\n", last_file, last_include
}

FNR == 1 {
    if (NR!=1) end_file()
    last_use = last_include = 0
    last_file = FILENAME
}

{
    # strip comments
    gsub("//.*", "")
}

is_include() {
    last_include = FNR
}

is_def_use() {
    last_use = FNR
}

END {
    end_file()
}
