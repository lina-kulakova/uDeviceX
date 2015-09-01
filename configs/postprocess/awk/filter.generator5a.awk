#!/usr/bin/awk -f

BEGIN {
    pre()
}

function pre() {
    printf "BEGIN {\n"
}

{
   sub(/#.*/, "")         # strip comments
}

!NF {
    next                  # strip empty lines
}

{
    key = $1
    val = $2
    printf "    %s = %s\n", key, val
}
    
