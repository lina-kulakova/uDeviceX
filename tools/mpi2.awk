#!/usr/bin/awk -f

# remove useless wrapper

BEGIN {
    name = "MPI_[A-Za-z0-9_]*"
    wspa = "[ \t]*"
    br   = "\\("
    
    pat = name wspa br
}

match(s = $0, pat) {
    i0 = index(s, "MC(")
    if (i0 == 0) {print; next} # no wrapper

    cnt = 0
    for (i = i0; i <= length(s); i++) {
	c = ch(i)
	if (c == "(") cnt++
	if (c == ")") {
	    cnt--
	    if (cnt == 0) {s = upd(i0, i); break}
	}
    }
    print s
}

{
    print
}

function upd(i, j,    ans, lo, hi) {
    lo = 1; hi = i - 1;
    ans = ans substr(s, 1, i - 1)
    ans = ans substr(s, i + length("MC("), j - i - length("MC("))
    ans = ans substr(s, j + 1)
    return ans
}

function ch(i) {return substr(s, i, 1)}
