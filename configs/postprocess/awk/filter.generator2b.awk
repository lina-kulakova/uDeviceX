#!/usr/bin/awk -f

# TEST: fg2a1
# ./filter.generator2a.awk test_data/test1.filter | ./filter.generator2b.awk > fg.out.txt
#
# TEST: fg2a2
# ./filter.generator2a.awk test_data/test2.filter | ./filter.generator2b.awk > fg.out.txt
#
# TEST: fg2a3
# ./filter.generator2a.awk test_data/test3.filter | ./filter.generator2b.awk > fg.out.txt
#
# TEST: fg2a4
# ./filter.generator2a.awk test_data/test4.filter | ./filter.generator2b.awk > fg.out.txt


function is_separator() {
    return substr(line, 1, 1)!=" "
}

function is_function(s) {
    return substr(line, 1, 1)==" "
}

function read_file(line) {
    while (getline line >= 1)
	arr[++max_pos] = line
}

function next_line(        ) {
    pos++
    if (pos>max_pos)
	return FA

    line = arr[pos]
    return SU
}

function put_back() {
    pos--
    line = arr[pos]    
}

BEGIN {
    SU = 1
    FA = 0
    
    read_file()
    
    while (1) {
	eat_separator()
	if (rc<1) break
	eat_function()
	if (rc<1) break	
    }
}


function eat_separator() {
    while (1) {
	rc = next_line()
	if (rc<1) return
	if (!is_separator()) {
	    put_back()
	    return
	}
    }
}

function eat_function(rc, body, max_ibody) {
    delete body
    
    while (1) {
	rc = next_line()
	if (rc<1) break
	if (!is_function()) {
	    put_back()
	    break
	}
	body[++max_ibody] = line
    }
    write_function(body, max_ibody)
}

function write_function(body, max_ibody,   i, aux) {
    ifunction++

    printf "function __fun__rc%d() {\n", ifunction

    for (i=1; i<=max_ibody-1; i++)
	printf "%s\n", body[i]


    aux = body[max_ibody]
    sub(/^ +/, "", aux)
    printf "   return %s\n",  aux
    
    printf "}\n\n"
}
