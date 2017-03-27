#include <stdio.h>
#include <random>

#define MAX_NP 100000 /* maximum number of particles */

#define dir "xyz"     /* output directory and file name */
#define fmt "%03ld.3d"

double xx[MAX_NP],  yy[MAX_NP], zz[MAX_NP],
      xxn[MAX_NP], yyn[MAX_NP], zzn[MAX_NP]; /* "next" particle position */
double vvx[MAX_NP], vvy[MAX_NP], vvz[MAX_NP]; /* velocity */
long type[MAX_NP];

long np; /* number of particles */
double xl, yl, zl; /* domain */
double xh, yh, zh;
double Lx, Ly, Lz;
double xc, yc, zc; /* center */

double vx0, vy0, vz0;
long   type0;

long  ts, ns;
double dt;

void init_vars() {
  np = 300; /* number of particles */
  xl = -10; yl = -10; zl = -10; /* domain */
  xh =  10; yh =  10; zh =  10;
  Lx = xh - xl; Ly = yh - yl; Lz = zh - zl;
  xc = 0.5*(xh + xl); yc = 0.5*(yh + yl); zc = 0.5*(zh + zl);

  vx0 = -2; vy0 = 0; vz0 = 0; /* initial velocity */
  type0 = 0;                  /* initial type  */
  ts = 0;    /* current time frame (0, 1, 2, ...) */
  ns =  100;   /* number of time steps to make */
  dt = 0.1;   /* time steps */

  system("mkdir " dir);
}


float rnd(float lo, float hi) {
  using namespace std;
  static random_device rd;
  static default_random_engine e(rd()) ;
  static uniform_real_distribution<> dist(0, 1);
  return dist(e) * (hi - lo) + lo;
}

void init_pos() {
    for (long ip = 0; ip < np; ip++) {
      xx[ip] = rnd(xl, xh); yy[ip] = rnd(yl, yh);  zz[ip] = rnd(zl, zh);
    }
}

void init_type() {
  for (long ip = 0; ip < np; ip++) type[ip] = type0;
}

void init_vel() {
    for (long ip = 0; ip < np; ip++) {
      vvx[ip] = vx0; vvy[ip] = vy0; vvz[ip] = vz0;
    }
}

void print_part0(FILE* fd) {
    for (long ip = 0; ip < np; ip++)
      fprintf(fd, "%g %g %g %ld\n", xx[ip], yy[ip], zz[ip], type[ip]);
}

void print_part() { /* sets and manage file name */
  char fn[2048];
  sprintf(fn, dir "/" fmt, ts);
  auto fd = fopen(fn, "w");
  print_part0(fd);
  fclose(fd);
}

void new_pos() {
    for (long ip = 0; ip < np; ip++) {
      xxn[ip] = xx[ip] + dt*vvx[ip];
      yyn[ip] = yy[ip] + dt*vvy[ip];
      zzn[ip] = zz[ip] + dt*vvz[ip];
    }
}

void upd_vel() {
  /*  do nothing: keep constant velocity */
}

double wrp(double r, double c, double L) { /* wrap back to the domain */
  auto dr = r - c;
  if      (2*dr >  L) return r - L;
  else if (2*dr < -L) return r + L;
  else                return r;
}

void pbc() { /* periodic boundary conditions */
    for (long ip = 0; ip < np; ip++) {
      xx[ip] = wrp(xx[ip], xc, Lx);
      yy[ip] = wrp(yy[ip], yc, Ly);
      zz[ip] = wrp(zz[ip], zc, Lz);
    }
}


void bounce() {
    for (long ip = 0; ip < np; ip ++) {
      auto x  = xx [ip],  y  = yy [ip], z  = zz [ip];
    }
}

void upd_pos() { /* new to old */
    for (long ip = 0; ip < np; ip ++) {
      xx[ip] = xxn[ip]; yy[ip] = yyn[ip]; zz[ip] = zzn[ip];
    }
}

void step() {
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
}

int main() {
  init();
  while (ts < ns) {
    print_part();
    upd();
  }
}
