#!/usr/bin/awk -f

# generate literate part

BEGIN {
    LITERATE_MARKER = "%%%"
}

function is_literate() {
    return substr($0, 1, 3)==LITERATE_MARKER
}

{
   sub(/#.*/, "")         # strip comments
}

!NF {
   next                  # strip empty lines    
}

is_literate() {
    has_body = 1
    sub(LITERATE_MARKER, "")
    if (substr($0, 1, 1)==" ")
	sub(" ", "")
    print
}

END {
    if (has_body)
	printf "\n"
}
