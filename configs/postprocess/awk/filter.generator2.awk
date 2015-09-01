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
    next                  # strip literate part
}


