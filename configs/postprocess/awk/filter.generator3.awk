#!/usr/bin/awk -f

# TEST: fg31
# ./filter.generator2a.awk test_data/test1.filter | ./filter.generator3.awk > fg.out.txt
#
# TEST: fg32
# ./filter.generator2a.awk test_data/test2.filter | ./filter.generator3.awk > fg.out.txt
#
# TEST: fg33
# ./filter.generator2a.awk test_data/test3.filter | ./filter.generator3.awk > fg.out.txt
#
# TEST: fg34
# ./filter.generator2a.awk test_data/test4.filter | ./filter.generator3.awk > fg.out.txt

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
    printf "\n"
    
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
    printf "     __rc%d = __fun__rc%d()\n", ifunction, ifunction
}
