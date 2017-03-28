__HD__ void   cycle(float *r) {
  if (D == Y) return;
  float r0[3] = {r[X], r[Y], r[Z]};
  if (D == X) { /* ZXY  = XYZ */
    r[Z] = r0[X]; r[X] = r0[Y]; r[Y] = r0[Z];
  } else {      /* YZX  = XYZ */
    r[Y] = r0[X]; r[Z] = r0[Y]; r[X] = r0[Z];
  }
}

__HD__ void uncycle(float *r) {
  if (D == Y) return;
  float r0[3] = {r[X], r[Y], r[Z]};
  if (D == X) { /* XYZ = ZXY */
    r[X] = r0[Z]; r[Y] = r0[X]; r[Z] = r0[Y];
  } else {      /* XYZ = YZX */
    r[X] = r0[Y]; r[Y] = r0[Z]; r[Z] = r0[X];
  }
}

__HD__ void rg2l(float *rg, /**/ float *rl) { /* global to local (position) */
  float rc[3] = {rcx, rcy, rcz};
  int c;
  for (c = 0; c < 3; c++)
    rl[c] = (rg[c] - rc[c])/rcyl;
  cycle(rl);
}

__HD__ void vg2l(float *vg, /**/ float *vl) { /* global to local (velocity) */
  int c;
  for (c = 0; c < 3; c++)
    vl[c] = vg[c] / rcyl;
  cycle(vl);
}

__HD__ void rl2g(float *rl, /**/ float *rg) { /* local to global (position) */
  float rc[3] = {rcx, rcy, rcz};
  int c;
  for (c = 0; c < 3; c++) rg[c] = rl[c];
  uncycle(rg);
  for (c = 0; c < 3; c++) rg[c] = rg[c] * rcyl + rc[c];
}

__HD__ void vl2g(float *vl, /**/ float *vg) { /* local to global (velocity) */
  int c;
  for (c = 0; c < 3; c++) vg[c] = vl[c];
  uncycle(vg);
  for (c = 0; c < 3; c++) vg[c] *= rcyl;
}

__HD__ bool inside_sc(float *r) {
  return r[X]*r[X] + r[Z]*r[Z] < 1;
}

__HD__ bool inside(float *rg) {
  float rl[3];
  rg2l(rg, rl);
  return inside_sc(rl);
}

__HD__ void vwall_sc(float *r, /**/ float *v) {
  float om = 0;
  v[X] = -om*r[Z]; v[Y] = 0; v[Z] = om*r[X];
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
__HD__ void wavg(float* r0, float* r1, float h, /**/ float* rh) {
  int c;
  for (c = 0; c < 3; c++) rh[c] = r0[c]*(1-h) + r1[c]*h;
}

__HD__ void bb_vel(float *v0, float *vw, /**/ float *vn) {
  int c;
  for (c = 0; c < 3; c++) vn[c] = 2*vw[c] - v0[c];
}

__HD__ void bb_pos(float *rw, float *vn, float h, /**/ float *rn) {
  int c;
  for (c = 0; c < 3; c++) rn[c] = rw[c] + vn[c]*(1 - h)*dt;
}

__HD__ int rescue(float *r, float* v) {
  float rmag, rnew, sc, rw[3], vw[3], vn[3];
  rmag = sqrt(r[X]*r[X] + r[Z]*r[Z]);
  rw[X] = r[X]/rmag; rw[Y] = r[Y]; rw[Z] = r[Z]/rmag;
  
  vwall_sc(rw, /**/ vw);
  bb_vel(v, vw, /**/ vn);
  rnew = 1 + (1 - rmag); sc = rnew/rmag;
  r[X] *= sc; r[Z] *= sc;

  return BB_RESCUE;
}

__HD__ int bb1(float *r0, float *v0, float h,
	       float *r1, float *v1) {
  float rw[3], vw[3], rn[3], vn[3]; /* wall position, new position,
				       new velocity */
  wavg(r0, r1, h, /**/ rw);
  vwall_sc(rw, /**/ vw);

  bb_vel(v0, vw,    /**/ vn);
  bb_pos(rw, vn, h, /**/ rn);

  if (inside_sc(rn)) return rescue(r1, v0);

  int c;
  for (c = 0; c < 3; c++) {v1[c] = vn[c]; r1[c] = rn[c];}
  return BB_NORMAL;
}

__HD__ int bb0(float *r0, float *v0,
	       float *r1, float *v1) {
  float r0x = r0[X],         r0z = r0[Z];
  float drx = r1[X] - r0[X], drz = r1[Z] - r0[Z];

  float a, k, c;
  a = drz*drz + drx*drx;
  k = r0z*drz + r0x*drx;
  c = r0z*r0z + r0x*r0x - 1;

  float h0, h1;
  int n = solve_half_quadratic(a, k, c, /**/ &h0, &h1);

  if (n > 0 && h0 > 0 && h0 < 1)
    return bb1(r0, v0, h0, /**/ r1, v1);

  if (n > 1 && h1 > 0 && h1 < 1)
    return bb1(r0, v0, h1, /**/ r1, v1);

  return rescue(r1, v0);
}

__HD__ int bb(float *r0_, float *v0_, /*inout*/ float *r1_, float *v1_) {
  float r0[3], v0[3], r1[3], v1[3];
  rg2l(r1_, r1); rg2l(r0_, r0);

  if (!inside_sc(r1)) return BB_NO;
  if ( inside_sc(r0)) return BB_R0_INSIDE;

  vg2l(v0_, v0); vg2l(v1_, v1);

  int rcode = bb0(r0, v0, /**/ r1, v1);
  if (rcode == BB_NO) return BB_NO;

  rl2g(r1, r1_);
  vl2g(v1, v1_);

  return rcode;
}
