#!/usr/bin/awk -f

# Generate the adjVert and adjVert2 from off file
# TEST: adj2.t1
#  ./adj2.awk ../src/rbc.off
# cat a[12].txt > a.out.txt

function init() {
    md = 7
    fn = ARGC < 2 ? "-" : ARGV[1]
}

function nl() { # next line
    getline < fn
}

function read_header() {
    nl() # OFF
    nl()
    nv = $1; nf = $2
}

function skip_vert(   iv) {
    for (iv = 0; iv < nv; iv++) nl()
}

function read_faces(   ifa, ib) {
    for (ifa = 0; ifa < nf; ifa++) {
	nl()
	ib = 2 # skip number of vertices per face
	ff0[ifa] = $(ib++); ff1[ifa] = $(ib++); ff2[ifa] = $(ib++);
    }
}

function read_off() {
    read_header()
    skip_vert()
    read_faces()
}

function init_a(   i) {
    for (i = 0; i < nv*md; i++)
	hx[i] = a1[i] = a2[i] = -1
}

function write_a(   i) {
    for (i = 0; i < nv*md; i++) {
	print a1[i] > "a1.txt"
	print a2[i] > "a2.txt"
    }
    close("a1.txt"); close("a2.txt");
}

function reg_edg(i, x, y) {
    i *= md
    while (hx[i] != -1) i++
    hx[i] = x; hy[i] = y
}

function init_edg(   ifa) {
    for (ifa = 0; ifa < nf; ifa++) {
	f0 = ff0[ifa]; f1 = ff1[ifa]; f2 = ff2[ifa]
	reg_edg(f0, f1, f2);
	reg_edg(f1, f2, f0);
	reg_edg(f2, f0, f1);	
    }
}

function nxt(i, x) {
    i *= md
    while (hx[i] != x) i++;
    return hy[i]
}

function gen_a10(i0,   c, fst) {
    lo = i0*md; hi = lo + md;
    mi = hx[lo]
    for (i = lo + 1; i < hi && hx[i] != -1; i++)
	if (hx[i] < mi) mi = hx[i]

    i = lo; c = mi
    do {
	c = nxt(i0, c0 = c)
	a1[i] = c0
	a2[i] = nxt(c, c0)
	i++
    }  while (c != mi)
    
}

function gen_a1() {
    for (i0 = 0; i0 < nv; i0++)
	gen_a10(i0)
}

BEGIN {
    init()
    read_off()

    init_a()
    init_edg()

    gen_a1()

    write_a()
}
