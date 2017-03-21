#include "hd.def.h"
#include "awall.h"

__HD__ void fun(float *a, float *b, /**/ float *ans) {
  enum {X, Y, Z};
  *ans = a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z];
}

__HD__ void   cycle(int D, float* R) {
  if (D == Y) return;
  float R0[3] = {R[X], R[Y], R[Z]};
  if (D == X) { /* ZXY  = XYZ */
    R[Z] = R0[X];
    R[X] = R0[Y];
    R[Y] = R0[Z];
  } else {      /* YZX  = XYZ */
    R[Y] = R0[X];
    R[Z] = R0[Y];
    R[X] = R0[Z];
  }
}

__HD__ void uncycle(int D, float* R) {
  if (D == Y) return;
  float R0[3] = {R[X], R[Y], R[Z]};
  if (D == X) { /* XYZ = ZXY */
    R[X] = R0[Z];
    R[Y] = R0[X];
    R[Z] = R0[Y];
  } else {      /* XYZ = YZX */
    R[X] = R0[Y];
    R[Y] = R0[Z];
    R[Z] = R0[X];
  }
}


__HD__ int bb0(float *R0, float *V0,
	       float *R1, float *V1,
	       float *dP, float *dL) {
  return BB_NO;
}

__HD__ int bb(float *R0_, float *V0_,
	      float *Rc , float rcyl, int D, /* center, radius, axis
						(X, Y, Z) of a
						cylinder */
	      float *R1_, float *V1_, /*in-out*/
	      float *dP , float *dL) /*out*/ {
  float R0[3], V0[3], R1[3], V1[3];
  int c;
  for (c = 0; c < 3; c++) { /* copy input */
    R0[c] = R0_[c]; V0[c] = V0_[c];
    R1[c] = R1_[c]; V1[c] = V1_[c];
  }

  for (c = 0; c < 3; c++) { /* scale and shift */
    R0[c] -= Rc[c]; R0[c] /= rcyl;
    /*          */  V0[c] /= rcyl;
    R1[c] -= Rc[c]; R1[c] /= rcyl;
    /*          */  V1[c] /= rcyl;
  }

#define  cy(R)   cycle(D, (R))
  cy(R0); cy(V0); cy(R1); cy(V1);

  int rc = bb0(R0, V0, /*io*/ R1, V1, /*o*/ dP, dL);
  if (rc == BB_NO) return BB_NO;

#define ucy(R) uncycle(D, (R))
  ucy(R1); ucy(V1);
  ucy(dP); ucy(dL);

  for (c = 0; c < 3; c++) { /* unshift and unscale */
    R1[c] *= rcyl; R1[c] += Rc[c];
    V1[c] *= rcyl; /*           */

    dP[c] *= rcyl;
    dL[c] *= rcyl*rcyl;
  }

  for (c = 0; c < 3; c++) { /* copy output */
    R0_[c] = R0[c]; V0_[c] = V0[c];
    R1_[c] = R1[c]; V1_[c] = V1[c];
  }

  return 0;
}
