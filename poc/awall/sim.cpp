#include <stdio.h>
#include <stdlib.h>

#include <string.h> /* memcpy */

#include "hd.def.h"
#include "awall.h"
#define dt 0.1

#define dir "3d"     /* output directory and file name format */
#define file_fmt "%05ld.3D"

#define fmt      "%+.12e"
#define bbox "bbox.vtk" /* output a file with simulation domain */

/* MAX number of particles */
#define MAX_N 3000
float rr0[3*MAX_N], rr1[3*MAX_N], vv0[3*MAX_N], vv1[3*MAX_N];
long type[MAX_N];

long n; /* number of particles */

float xl, yl, zl; /* domain */
float xh, yh, zh;
float Lx, Ly, Lz;
float  xc, yc, zc; /* center */

 /* initial velocity */
float vx0, vy0, vz0;
long  type0;

long  ts, ns, dfrq;

/* Langevin equation parameter: dissipation and temperature */
#define la 0.1
#define  T 1.0

enum {X, Y, Z};

/* copy all particle */
#define copy(A, B)  memcpy((A), (B), 3*n*sizeof((B)[0]));

/* copy one particle */
#define copy0(A, B) memcpy((A), (B),  3*sizeof((B)[0]));

void init_vars() {
  n  = 3000; /* number of particles */
  xl = -10; yl = -10; zl = -10; /* domain */
  xh =  10; yh =  10; zh =  10;
  Lx = xh - xl; Ly = yh - yl; Lz = zh - zl;
  xc = 0.5*(xh + xl); yc = 0.5*(yh + yl); zc = 0.5*(zh + zl);

  vx0 = 1.0; vy0 = 0; vz0 = 0; /* initial velocity */
  type0 = 0;                  /* initial type  */
  ts = 0;     /* current time frame (0, 1, 2, ...) */
  ns =  1000;   /* number of time steps to make */
  dfrq = 10;  /* dump every `dfrq' time steps */

  system("mkdir -p " dir);
}

float rnd(float lo, float hi) {
  return drand48()*(hi - lo) + lo;
}

void init_pos() {
  for (long ip = 0; ip < n; ip++) {
    float *r0 = &rr0[3*ip];
    r0[X] = rnd(xl, xh); r0[Y] = rnd(yl, yh);  r0[Z] = rnd(zl, zh);
  }
}

bool inside_main(float *r) {
  return inside(r);
}

void filter_pos() { /* updates `n' */
  long ip = 0, jp = 0;
  for (/*  */ ; ip < n; ip++) {
    if (inside_main(rr0 + 3*ip)) continue;
    copy0(rr0 + 3*jp, rr0 + 3*ip);
    jp++;
  }
  n = jp;
}

void init_type() {
  for (long ip = 0; ip < n; ip++) type[ip] = type0;
}

void init_vel() {
    for (long ip = 0; ip < n; ip++) {
      float *v0 = &vv0[3*ip];
      v0[X] = vx0; v0[Y] = vy0; v0[Z] = vz0;
    }
}

void print_bbox() {
#define pr(...) fprintf (fd, __VA_ARGS__)
  FILE* fd = fopen(bbox, "w");
  pr("# vtk DataFile Version 3.0\n");
  pr("vtk output\n");
  pr("ASCII\n");
  pr("DATASET STRUCTURED_POINTS\n");
  pr("DIMENSIONS 2 2 2\n");
  pr("ORIGIN %g %g %g\n" , xl, yl, zl);
  pr("SPACING %g %g %g\n", Lx, Ly, Lz);
  fclose(fd);
#undef pr
}

void print_part0(FILE* fd) {
  #define s " "
  fprintf(fd, "x y z type\n");
  for (long ip = 0; ip < n; ip++) {
    float *r0 = &rr0[3*ip];
    fprintf(fd, fmt s fmt s fmt s "%ld\n", r0[X], r0[Y], r0[Z], type[ip]);
  }
  #undef s
}

void print_part() { /* sets and manage file name */
  char fn[BUFSIZ];
  sprintf(fn, dir "/" file_fmt, ts);
  FILE* fd = fopen(fn, "w");
  print_part0(fd);
  fclose(fd);
}

void new_pos() {
    for (long ip = 0; ip < n; ip++) {
      float *r0 = &rr0[3*ip], *r1 = &rr1[3*ip], *v0 = &vv0[3*ip];
      r1[X] = r0[X] + dt*v0[X];
      r1[Y] = r0[Y] + dt*v0[Y];
      r1[Z] = r0[Z] + dt*v0[Z];
    }
}

float lang(float v) { /* Langevin equation RHS */
  v -= dt * la * v;
  v += dt * rnd(-T, T);
  return v;
}

void new_vel() {
  for (long ip = 0; ip < n; ip++) {
    float *v0 = &vv0[3*ip], *v1 = &vv1[3*ip];
    v1[X] = lang(v0[X]); v1[Y] = lang(v0[Y]); v1[Z] = lang(v0[Z]);
  }
}

float wrp(float r, float c, float L) { /* wrap back to the domain */
  float dr = r - c;
  if      (2*dr >  L) return r - L;
  else if (2*dr < -L) return r + L;
  else                return r;
}

void pbc() { /* periodic boundary conditions */
  for (long ip = 0; ip < n; ip++) {
    float *r1 = &rr1[3*ip];
    r1[X] = wrp(r1[X], xc, Lx);
    r1[Y] = wrp(r1[Y], yc, Ly);
    r1[Z] = wrp(r1[Z], zc, Lz);
  }
}

void bounce() {
  for (long ip = 0; ip < n; ip ++) {
    float *r0 = &rr0[3*ip], *r1 = &rr1[3*ip];
    float *v0 = &vv0[3*ip], *v1 = &vv1[3*ip];
    int code = bb(r0, v0, r1, v1);
  }
}

void step() { /* simulation step */
  new_pos();
  new_vel();

  bounce();  /* bouncing back */
  pbc();     /* periodic BC */

  copy(rr0, rr1); copy(vv0, vv1);
  ts += 1; /* update timestep */
}

void init() {
  init_vars(); /* variables */
  init_pos(); /* particle positions */
  filter_pos(); /* kill particles inside the wall */

  init_type(); /*  ...      types */
  init_vel();  /* ...      velocity */

  print_bbox();  /* dump a file with simulation domain */
}

int main() {
  init();
  while (ts <= ns) {
    if (ts % dfrq == 0) print_part();
    step();
  }
}
