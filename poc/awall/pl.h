__HD__ void rg2l(float *rg, /**/ float *rl) { /* global to local (position) */
  float rc[3] = {rcx, rcy, rcz};
  int c;
  for (c = 0; c < 3; c++) rl[c] = rg[c] - rc[c];
}

__HD__ void rl2g(float *rl, /**/ float *rg) { /* local to global (position) */
  float rc[3] = {rcx, rcy, rcz};
  int c;
  for (c = 0; c < 3; c++) rg[c] = rl[c] + rc[c];
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
  float rx = r[X], ry = r[Y], rz = r[Z];
  float h, b = nz*rz+ny*ry+nx*rx;
  h = - b;
  
  float rw[3] = {rx+h*nx, ry+h*ny, rz+h*nz};
  float vw[3], vn[3];
  vwall_sc(rw, /**/ vw);  
  bb_vel(v, vw, /**/ vn);
  
  r[X] = rx+2*h*nx;
  r[Y] = ry+2*h*ny;
  r[Z] = rz+2*h*nz;

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
  float r0x = r0[X], r0y = r0[Y], r0z = r0[Z];

  float drx = r1[X] - r0[X];
  float dry = r1[Y] - r0[Y];
  float drz = r1[Z] - r0[Z];

  float a = drz*nz+dry*ny+drx*nx;
  if (a == 0) return rescue(r1, v0);
  
  float b = nz*r0z+ny*r0y+nx*r0x;
  float h = - b / a;

  if (h > 0 && h < 1)
    return bb1(r0, v0, h, /**/ r1, v1);

  return rescue(r1, v0);
}

__HD__ int bb(float *r0_, float *v0, /*inout*/ float *r1_, float *v1) {
  float r0[3], r1[3];
  rg2l(r1_, r1); rg2l(r0_, r0);

  if (!inside_sc(r1)) return BB_NO;
  if ( inside_sc(r0)) return BB_R0_INSIDE;

  int rcode = bb0(r0, v0, /**/ r1, v1);
  if (rcode == BB_NO) return BB_NO;

  rl2g(r1, r1_);

  return rcode;
}
