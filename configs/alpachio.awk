#!/usr/bin/awk -f

BEGIN {
    FS = "//="
}

NF == 2 {
    print $2, 
}
