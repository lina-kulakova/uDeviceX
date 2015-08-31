#!/usr/bin/awk -f

# output two columns `x' and `id' sorted by `x'

BEGIN {
    id_idx = 1
    x_idx = 4
}

$0 == "Atoms" {
    getline # skip empty line

    pipe = "sort -g "
    while (getline >0  && NF) {
	id = $(id_idx) ; x = $(x_idx)
	print x, id |  pipe
    }
    close(pipe)
}
