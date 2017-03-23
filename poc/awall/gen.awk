#!/usr/bin/awk -f

# Test data generator for wall

function rnd(lo, hi) {return lo + (hi - lo)*rand()}

function ini() {
    lo = -2
    hi =  2

    vlo = -1
    vhi =  1

    rlo = -0.2
    rhi =  0.2 # random shift

    pi = 3.1415926
    
    n = 1000
}

function gen_pos_cyl(   i, x, y, z) {
    for (i = 0; i < n; i++) {
	phi = rnd(0, 2*pi)
	x = cos(phi)    + rnd(rlo, rhi)
	y = sin(phi)    + rnd(rlo, rhi)
	z = rnd(lo, hi)
	xx[i] = x; yy[i] = y; zz[i] = z
    }
}

function gen_pos_box(   i, x, y, z) {
    for (i = 0; i < n; i++) {
	x = rnd(lo, hi); y = rnd(lo, hi); z = rnd(lo, hi);
	xx[i] = x; yy[i] = y; zz[i] = z
    }
}

function gen_vel(   i, vx, vy, vz) {
    for (i = 0; i < n; i++) {
	vx = rnd(vlo, vhi); vy = rnd(vlo, vhi); vz = rnd(vlo, vhi);
	vvx[i] = vx; vvy[i] = vy; vvz[i] = vz
    }
}

function dump(   i, x, y, z, vx, vy, vz, cmd, rc) {
    for (i = 0; i < n; i++) {
	 x =  xx[i];  y =  yy[i];  z =  zz[i]
	vx = vvx[i]; vy = vvy[i]; vz = vvz[i]
	cmd = sprintf("./main %s %s %s  %s %s %s", x, y, z, vx, vy, vz)
	rc = system(cmd)
	if (rc) exit;
    }
}

BEGIN {
    ini()
    gen_pos_cyl()
    gen_vel()
    dump()
}
