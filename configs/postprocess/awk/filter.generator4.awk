#!/usr/bin/awk -f

# TEST: fg41
# ./filter.generator2a.awk test_data/test1.filter | ./filter.generator4.awk > fg.out.txt
#
# TEST: fg42
# ./filter.generator2a.awk test_data/test2.filter | ./filter.generator4.awk > fg.out.txt
#
# TEST: fg43
# ./filter.generator2a.awk test_data/test4.filter | ./filter.generator4.awk > fg.out.txt
#
# TEST: fg44
# ./filter.generator2a.awk test_data/test4.filter | ./filter.generator4.awk > fg.out.txt

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
    TAB  = "       "
    
    SU = 1
    FA = 0
    
    read_file()
    pre()
    
    while (1) {
	eat_separator()
	if (rc<1) break
	eat_function()
	if (rc<1) break	
    }

}


function eat_separator() {
    TAB = "         "
    while (1) {
	rc = next_line()
	if (rc<1) break
	if (!is_separator()) {
	    put_back()
	    break
	}
	printf TAB line "                  \\\n"
    }
}

function eat_function(rc) {
    while (1) {
	rc = next_line()
	if (rc<1) break
	if (!is_function()) {
	    put_back()
	    break
	}
    }
    
    ifunction++
    printf TAB "__rc" "%d"  "                  \\\n", ifunction
}

END {
    post()
}

function pre() {
    printf "\n"
    printf "   __rc = \\\n"
    printf "   ( \\\n"
}

function post() {
    printf "   )\n"
    printf "\n"
}
