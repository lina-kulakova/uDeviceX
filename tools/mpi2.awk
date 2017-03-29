#!/usr/bin/awk -f

BEGIN {
    fu = "MPI_[A-Za-z_]*"
    sp = "[\n\t]*"
    obr = "\\(" # opening
    cbr = "\\)" # closing
    any = "."

#    pat = fu sp br
}

{
    sep = NR == 1 ? "" : ORS
    f = f sep $0
}

function print_name() { # prints all MPI function names (kill 'f')
    print substr(f, RSTART, RLENGTH)
    f   = substr(f, RSTART + RLENGTH)
}


END {
    while (nxt(fu)) {
	fn = TOK
	nxt("^" sp) # eat spaces
	rc = nxt("^" obr)
	if (!rc) continue  # no '(' => not a function

	cnt = 1 # opening-closing
	arg = "" # arguments of the call
	while (length(f)) {
	    nxt("^" obr) || nxt("^" cbr) || nxt("^" any)
	    if (TOK == ")") cnt--
	    if (TOK == "(") cnt++
	    if (cnt == 0) break
	    arg = arg TOK
	}
	print fn
	print "(" arg ")"
    };
}

function ch(i) {return substr(f, i, 1)} # charachter

function nxt(pat, rc) { # set TOK, advince in f
    rc = match(f, pat)
    if (rc == 0) return rc

    TOK = substr(f, RSTART, RLENGTH)
    f =   substr(f, RSTART + RLENGTH)
    return rc
}
