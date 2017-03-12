#!/usr/bin/awk -f

# Generate the adjVert and adjVert2 from off file

function init() {
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
	a1[i] = a[i] = -1
}

function write_a(   i) {
    for (i = 0; i < nv*md; i++) {
	print a1[i] > "a1.txt"
	print a2[i] > "a2.txt"
    }
    close("a1.txt"); close("a2.txt");
}

function reg_edg(a, b, c) {
    if ((a, b) in nxt) return
    if ((b, a) in nxt) return
    nxt[a,b] = prv[b,a] = c

    ee0[ne] = a; ee1[ne] = b; ne++
    deg[a]++; deg[b]++ # degree of a vertice
}

function init_edg(   ifa) {
    ne = 0 # number of edges
    for (ifa = 0; ifa < nf; ifa++) {
	f0 = ff0[ifa]; f1 = ff1[ifa]; f2 = ff2[ifa]
	reg_edg(f0, f1, f2);
	reg_edg(f2, f0, f1);
	reg_edg(f1, f2, f0);
    }
}

function max_deg(   iv, e0, e1) {
    md = 0
    for (iv = 0; iv < nv; iv++)
	if (deg[iv] > md) md = deg[iv]
    return md
}

function gen_a1(   iv) {
    for (iv = 0; iv < nv; iv++) {
	print iv
    }
}

BEGIN {
    init()
    read_off()

    init_edg()
    max_deg()
    init_a()

    gen_a1()

    write_a()
    print "md:", md
}
