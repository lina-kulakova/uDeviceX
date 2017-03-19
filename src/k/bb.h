namespace k_bb { /* bounce back */
    enum {X, Y, Z};

  __device__ void bounce_old(float currsdf,
			     float  *xp, float  *yp, float  *zp,
			     float *vxp, float *vyp, float *vzp) {
    float x = *xp,  y = *yp, z = *zp;
    float vx = *vxp, vy = *vyp, vz = *vzp;

    float x0 = x - vx*dt, y0 = y - vy*dt, z0 = z - vz*dt;
    /*
      Bounce back (stage I)

      Find wall position (sdf(wall) = 0): make two steps of Newton's
      method for the equation phi(t) = 0, where phi(t) = sdf(rr(t))
      and rr(t) = [x + vx*t, y + vy*t, z + vz*t]. We are going back
      and `t' is in [-dt, 0].

      dphi = v . grad(sdf). Newton step is t_new = t_old - phi/dphi

      Give up if dsdf is small. Cap `t' to [-dt, 0].
    */
#define rr(t) make_float3(x + vx*t, y + vy*t, z + vz*t)
#define small(phi) (fabs(phi) < 1e-6)
    float3 r, dsdf; float phi, dphi, t = 0;
    r = rr(t); phi = currsdf;
    dsdf = k_sdf::ugrad_sdf(r.x, r.y, r.z);
    dphi = vx*dsdf.x + vy*dsdf.y + vz*dsdf.z; if (small(dphi)) goto giveup;
    t -= phi/dphi;                            if (t < -dt) t = -dt; if (t > 0) t = 0;

    r = rr(t); phi = k_sdf::sdf(r.x, r.y, r.z);
    dsdf = k_sdf::ugrad_sdf(r.x, r.y, r.z);
    dphi = vx*dsdf.x + vy*dsdf.y + vz*dsdf.z; if (small(dphi)) goto giveup;
    t -= phi/dphi;                            if (t < -dt) t = -dt; if (t > 0) t = 0;
#undef rr
#undef small
  giveup:
    /* Bounce back (stage II)
       change particle position and velocity
    */
    float xw = x + t*vx, yw = y + t*vy, zw = z + t*vz; /* wall position */
    x += 2*t*vx; y += 2*t*vy; z += 2*t*vz; /* bouncing relatively to
					      the wall */
    k_wvel::bounce_vel(xw, yw, zw, /**/ vxp, vyp, vzp);
    if (k_sdf::sdf(x, y, z) >= 0) {*xp = x0; *yp = y0; *zp = z0;}
    else                          {*xp =  x; *yp =  y; *zp =  z;}
  }

  __device__ void lin(float *R1, float *V, float t, /**/ float *Rt) {
    /* weighted averaged */
    int c;
    for (c = 0; c < 3; c++) Rt[c] = R1[c]  + V[c]*(t - dt);
  }

   /* bounce postion in-place */
  __device__ void bounce_pos(float *Rw, /**/ float *R) {
    int c;
    for (c = 0; c < 3; c++) R[c] = Rw[c] + (Rw[c] - R[c]);
  }

  __device__ void bounce_cyl2(float *R1, float *V, float t) {
    float Rw[3]; lin(R1, V, t, /**/ Rw); /* wall position */
    bounce_pos(Rw, /**/ R1);
    k_wvel::bounce_vel(Rw[X], Rw[Y], Rw[Z], /**/ &V[X], &V[Y], &V[Z]);
  }

  __device__ void bounce_cyl1(float *R1, float *V) {
    float R1x = R1[X], R1z = R1[Z], \
           Vx =  V[X],  Vz =  V[Z];

    float a, b, c, D, sqD, s, t, eps = 1e-16;
    a = pow(Vz,2)+pow(Vx,2);
    if (fabs(a) < eps)  return;

    b = 2*(R1z*Vz+R1x*Vx);
    c = pow(R1z,2)+pow(R1x,2)-1;
    D = pow(b,2)-4*a*c;
    if (D < 0)          return;

    sqD = sqrt(D);
    s = (-b - sqD)/(2*a); t = s + dt;
    if      (t > 0 && t < dt) {
      bounce_cyl2(R1, V, t);
      return;
    }

    s = (-b + sqD)/(2*a); t = s + dt;
    if (t > 0 && t < dt)
      bounce_cyl2(R1, V, t);

  }

  __device__ void bounce_cyl0(float *R1, float *V) {
    /* move to the center of the cylinder and normalize by its radius */
    int c;
    float* RC = glb::r0, rcyl = 0.25, r;

    for (c = 0; c < 3; c++) {
      r = rcyl*glb::lg[c];
      R1[c] -= RC[c]; R1[c] /= r;
      /*           */  V[c] /= r;
    }

    bounce_cyl1(R1, V);

    for (c = 0; c < 3; c++) {
      r = rcyl*glb::lg[c];
      R1[c] *= r; R1[c] += RC[c];
      V [c] *= r;
    }
  }

  __device__ void bounce0(float sdf0,
			  float *R1, float *V) {
#ifdef acyl
    bounce_cyl0(R1, V);
    return;
#endif
    bounce_old(sdf0, &R1[X], &R1[Y], &R1[Z],  &V[X], &V[Y], &V[Z]);
  }

  __global__ void bounce(Particle *pp, int n) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    float *r = pp[pid].r, *v = pp[pid].v;
    float sdf0 = k_sdf::sdf(r[X], r[Y], r[Z]);
    if (sdf0 >= 0)
      bounce0(sdf0, r, v); /* dispatch */
  }
}  /* namespace k_bb */
