#include "hd.def.h"
#include "math.h"
#include "stdio.h"

#include "awall.h"

#undef  X
//#define X 0

#define X 0
#define Y 1
#define Z 2

#define D Z
#define dt 0.1
#define Rcx 0
#define Rcy 5.0
#define Rcz 0
#define rcyl 8.0

float Rw[3]; /* make it global so caller can check it */

__HD__ void   cycle(float* R) {
  if (D == Y) return;
  float R0[3] = {R[X], R[Y], R[Z]};
  if (D == X) { /* ZXY  = XYZ */
    R[Z] = R0[X]; R[X] = R0[Y]; R[Y] = R0[Z];
  } else {      /* YZX  = XYZ */
    R[Y] = R0[X]; R[Z] = R0[Y]; R[X] = R0[Z];
  }
}

__HD__ void uncycle(float* R) {
  if (D == Y) return;
  float R0[3] = {R[X], R[Y], R[Z]};
  if (D == X) { /* XYZ = ZXY */
    R[X] = R0[Z]; R[Y] = R0[X]; R[Z] = R0[Y];
  } else {      /* XYZ = YZX */
    R[X] = R0[Y]; R[Y] = R0[Z]; R[Z] = R0[X];
  }
}

__HD__ bool inside_sc(float *R) {
  return R[X]*R[X] + R[Z]*R[Z] < 1;
}

__HD__ bool inside(float *R) {
  float Rs[3], Rc[3] = {Rcx, Rcy, Rcz};
  int c;
  for (c = 0; c < 3; c++) {
    Rs[c] = R[c]; Rs[c] -= Rc[c]; Rs[c] /= rcyl;
  }
  cycle(Rs);
  return inside_sc(Rs);
}

__HD__ void vwall_sc(float *R, /**/ float *V) {
  float om = 0;
  V[X] = -om*R[Z]; V[Y] = 0; V[Z] = om*R[X];
}

__HD__ int solve_half_quadratic0(float k, float c, float *x0, float *x1) {
  /* solve x^2 + 2*k*x + c = 0 */
  float DD = k*k - c;
  if (DD > 0) {
    if (k == 0) {
      float r = sqrtf(-c);
      *x0 = -r; *x1 = r; return 2;
    } else {
      float sgnk = k > 0 ? 1 : -1;
      float r1 = -(k + sgnk*sqrt(DD));
      float r2 = c/r1;
      if (r1 < r2) {*x0 = r1; *x1 = r2;}
      else         {*x0 = r2; *x1 = r1;}
      return 2;
    }
  } else if (DD == 0) {
    *x0 = *x1 = -k; return 2;
  } else {
    return 0;
  }
}

__HD__ int solve_half_quadratic(float a, float k, float c, /**/ float *x0, float * x1) {
  /* solve "half quadratic*" equation; returns number of roots
     a*x^2 + 2*k*x + c = 0 */
  if (a == 0) {
    if (k == 0) return 0;
    else        {*x0 = -0.5*c/k; return 1;}
  }
  return solve_half_quadratic0(k/a, c/a, x0, x1);
}

/* weighted averaged */
__HD__ void wavg(float* R0, float* R1, float h, /**/ float* Rh) {
  int c;
  for (c = 0; c < 3; c++) Rh[c] = R0[c]*(1-h) + R1[c]*h;
}

__HD__ void bb_vel(float *V0, float *Vw, /**/ float *Vn) {
  int c;
  for (c = 0; c < 3; c++) Vn[c] = 2*Vw[c] - V0[c];
}

__HD__ void bb_pos(float *Rw, float *Vn, float h, /**/ float *Rn) {
  int c;
  for (c = 0; c < 3; c++) Rn[c] = Rw[c] + Vn[c]*(1 - h)*dt;
}

__HD__ int rescue(float *R, float* V) {
  float rmag, rnew, sc, /* Rw[3], */ Vw[3], Vn[3];
  rmag = sqrt(R[X]*R[X] + R[Z]*R[Z]);
  Rw[X] = R[X]/rmag; Rw[Y] = R[Y]; Rw[Z] = R[Z]/rmag;
  vwall_sc(Rw, /**/ Vw);

  bb_vel(V, Vw, /**/ Vn);
  rnew = 1 + (1 - rmag); sc = rnew/rmag;
  R[X] *= sc; R[Z] *= sc;

  return BB_RESCUE;
}

__HD__ int bb1(float *R0, float *V0, float h,
	       float *R1, float *V1) {
  float /* Rw[3], */ Vw[3], Rn[3], Vn[3]; /* wall position, new position,
				      new velocity */
  wavg(R0, R1, h, /**/ Rw);
  vwall_sc(Rw, /**/ Vw);

  bb_vel(V0, Vw,    /**/ Vn);
  bb_pos(Rw, Vn, h, /**/ Rn);

  if (inside_sc(Rn)) return rescue(R1, V0);

  int c;
  for (c = 0; c < 3; c++) {V1[c] = Vn[c]; R1[c] = Rn[c];}
  return BB_NORMAL;
}

__HD__ int bb0(float *R0, float *V0,
	       float *R1, float *V1) {
  float R0x = R0[X],         R0z = R0[Z];
  float dRx = R1[X] - R0[X], dRz = R1[Z] - R0[Z];

  float a, k, c;
  a = dRz*dRz + dRx*dRx;
  k = R0z*dRz + R0x*dRx;
  c = R0z*R0z + R0x*R0x - 1;

  float h0, h1;
  int n = solve_half_quadratic(a, k, c, /**/ &h0, &h1);

  if (n > 0 && h0 > 0 && h0 < 1)
    return bb1(R0, V0, h0, /**/ R1, V1);

  if (n > 1 && h1 > 0 && h1 < 1)
    return bb1(R0, V0, h1, /**/ R1, V1);

  return rescue(R1, V0);
}

__HD__ void g2l(float *rg, /**/ float *rl) { /* global to local */
  
}

__HD__ void l2g(float *rl, /**/ float *rg) { /* local to global */
  
}

__HD__ int bb(float *R0_, float *V0_,
	      /*inout*/
	      float *R1_, float *V1_) {
  int c;
  float R0[3], V0[3], R1[3], V1[3], Rc[3] = {Rcx, Rcy, Rcz};
  
  for (c = 0; c < 3; c++) { /* copy, shif, scale */
    R1[c] = R1_[c]; R1[c] -= Rc[c]; R1[c] /= rcyl;
    R0[c] = R0_[c]; R0[c] -= Rc[c]; R0[c] /= rcyl;
  }
  #define  cy(R) cycle(R)
  cy(R1); cy(R0);

  // TODO: check sdf
  if (!inside_sc(R1)) return BB_NO;
  if ( inside_sc(R0)) return BB_RO_INSIDE;

  for (c = 0; c < 3; c++) {
    V0[c] = V0_[c]; V0[c] /= rcyl;
    V1[c] = V1_[c]; V1[c] /= rcyl;
  }
  cy(V0); cy(V1);

  int rc = bb0(R0, V0, /**/ R1, V1);
  if (rc == BB_NO) return BB_NO;

  #define ucy(R) uncycle(R)
  ucy(R1); ucy(V1);
  for (c = 0; c < 3; c++) { /* unscale, unshift and copy */
    R1[c] *= rcyl; R1[c] += Rc[c]; R1_[c] = R1[c];
    V1[c] *= rcyl;               ; V1_[c] = V1[c];
  }
  return rc;
}
