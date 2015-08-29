function print_var_def(   nn, arr, i) {
    nn = split(parameters_line, arr, FIELD_SEP)
    for (i = 1; i<=nn; i+=2) {
	key=arr[i]
	val=arr[i+1]
	printf tab "%s = %s\n", key, val
    }
}

BEGIN {
    FIELD_SEP = "_"
    tab       = "    "
    parameters_line = ARGV[2]
    ARGV[2] = ""
}

function is_parameter() {
    return substr($0, 1, 1)=="="
}

# remove "="
function norm_name(s) {
    sub("^=", "", s)
    return s
}

{
   sub(/#.*/, "")         # strip comments
}

!NF {                    # skip empty lines
    next
}

is_parameter() {        # open a new parameter
    if (name) end_name()
    name = norm_name($1)
    start_name()
    next
}

name {
    process_name()
}

function end_name(   return_line, end_line) {
    return_line = sprintf(tab "return %s", last ? last : name )
    end_line    = sprintf("}\n")
    body = body RS return_line RS end_line
    body_arr[++iname] = body
    name_arr[iname  ] = name
}

function start_name() {
    last = ""
    body = sprintf("function __fun_%s() {", name)
}

function process_name() {
    if (last)
	body = body RS tab last
    last = $0
}

function print_ans(i, name, body) {
    for (i=1; i<=iname; i++) {
	name = name_arr[i]
	printf tab"__ans = __ans __sep() \"%s\" FIELD_SEP __fun_%s()\n", name, name
    }
}

function print_fun_def(i) {
    for (i=1; i<=iname; i++) {
	printf "\n"
	body = body_arr[i]
	printf "%s", body
    }
}

function begin_pre() {
    printf "function __sep() {\n"
    printf "    return __counter++ > 0 ? FIELD_SEP : \"\"\n"
    printf "}\n"
    printf "\n"
    printf "BEGIN {\n"
    printf "    FIELD_SEP=\"_\"\n"
    printf "\n"
}

function begin_post() {
    printf "\n"
    printf "    print __ans\n"
    printf "}\n"
}

END {
    end_name()
    begin_pre()
    
    print_var_def()
    printf "\n"
    print_ans()

    begin_post()
    
    print_fun_def()
}

