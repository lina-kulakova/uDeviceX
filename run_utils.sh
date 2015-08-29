#!/bin/bash
# Helper utils to run on remote host

# local top
ltop=$(git rev-parse --show-toplevel)

# current directory relative to the ltop
lcwd=$(git ls-files --full-name "${script_name}" | xargs dirname)


function msg() {
    printf "(run_utils) $@"  > "/dev/stderr"
}

# execute command remotely with (c)urrent directory: `local' 
function rc () {
    msg "(%s) execute on %s :  %s\n" "${rpath}/${default_dir}/${lcwd}" "${rhost}" "$@"
    ssh "${rhost}" "cd ${rpath}/${default_dir}/${lcwd} ; $@"
}

# execute command remotely with current directory: (t)op of git
function rt () {
    msg "(%s) execute on %s :  %s\n" "${rpath}/${default_dir}" "${rhost}" "$@"
    ssh "${rhost}" "cd ${rpath}/${default_dir} ; $@"
}
