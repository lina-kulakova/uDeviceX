#include <stdio.h>
#include <random>

#define dir "3d"     /* output directory and file name */
#define fmt "%03ld.3D"

#define bbox "bbox.vtk" /* simulation domain */

/* number of particles */
#define n 300
float xx[n],  yy[n], zz[n],
      xxn[n], yyn[n], zzn[n]; /* "next" particle position */
float vvx[n], vvy[n], vvz[n]; /* velocity */
long type[n];

float xl, yl, zl; /* domain */
float xh, yh, zh;
float Lx, Ly, Lz;
float  xc, yc, zc; /* center */

float vx0, vy0, vz0;
long   type0;

long  ts, ns;
float dt;

void init_vars() {
  xl = -10; yl = -10; zl = -10; /* domain */
  xh =  10; yh =  10; zh =  10;
  Lx = xh - xl; Ly = yh - yl; Lz = zh - zl;
  xc = 0.5*(xh + xl); yc = 0.5*(yh + yl); zc = 0.5*(zh + zl);

  vx0 = -2; vy0 = 0; vz0 = 0; /* initial velocity */
  type0 = 0;                  /* initial type  */
  ts = 0;    /* current time frame (0, 1, 2, ...) */
  ns =  10;   /* number of time steps to make */
  dt = 0.1;

  system("mkdir -p " dir);
}


float rnd(float lo, float hi) {
  return drand48()*(hi - lo) + lo;
}

void init_pos() {
    for (long ip = 0; ip < n; ip++) {
      xx[ip] = rnd(xl, xh); yy[ip] = rnd(yl, yh);  zz[ip] = rnd(zl, zh);
    }
}

void init_type() {
  for (long ip = 0; ip < n; ip++) type[ip] = type0;
}

void init_vel() {
    for (long ip = 0; ip < n; ip++) {
      vvx[ip] = vx0; vvy[ip] = vy0; vvz[ip] = vz0;
    }
}

void print_bbox() {
#define pr(...) fprintf (fd, __VA_ARGS__)
  auto fd = fopen(bbox, "w");
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
  fprintf(fd, "x y z type\n");
  for (long ip = 0; ip < n; ip++)
    fprintf(fd, "%g %g %g %ld\n", xx[ip], yy[ip], zz[ip], type[ip]);
}

void print_part() { /* sets and manage file name */
  char fn[BUFSIZ];
  sprintf(fn, dir "/" fmt, ts);
  auto fd = fopen(fn, "w");
  print_part0(fd);
  fclose(fd);
}

void new_pos() {
    for (long ip = 0; ip < n; ip++) {
      xxn[ip] = xx[ip] + dt*vvx[ip];
      yyn[ip] = yy[ip] + dt*vvy[ip];
      zzn[ip] = zz[ip] + dt*vvz[ip];
    }
}

void upd_vel() {
  /*  do nothing: keep constant velocity */
}

float wrp(float r, float c, float L) { /* wrap back to the domain */
  auto dr = r - c;
  if      (2*dr >  L) return r - L;
  else if (2*dr < -L) return r + L;
  else                return r;
}

void pbc() { /* periodic boundary conditions */
    for (long ip = 0; ip < n; ip++) {
      xx[ip] = wrp(xx[ip], xc, Lx);
      yy[ip] = wrp(yy[ip], yc, Ly);
      zz[ip] = wrp(zz[ip], zc, Lz);
    }
}

void bounce() {
    for (long ip = 0; ip < n; ip ++) {
      
    }
}

void upd_pos() { /* new to old */
    for (long ip = 0; ip < n; ip ++) {
      xx[ip] = xxn[ip]; yy[ip] = yyn[ip]; zz[ip] = zzn[ip];
    }
}

void step() { /* simulation step */
  new_pos(); /* get new position in `[xyz]n' */
  bounce(); /* bouncing back */
  upd_pos(); /* [xyz] = [xyz]n */
  pbc();     /* periodic BC */
  upd_vel(); /* get new velocity */
  ts += 1; /* update timestep */
}

void init() {
  init_vars(); /* variables */
  init_pos(); /* particle positions */
  init_type(); /*  ...      types */
  init_vel();  /* ...      velocity */

  print_bbox();  /* dump a file with simulation domain */
}

int main() {
  init();
  while (ts < ns) {
    print_part();
    step();
  }
}
