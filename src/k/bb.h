namespace k_bb { /* bounce back */
  __device__ void bounce0(float currsdf,
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

  __global__ void bounce(Particle *pp, int n) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    float *r = pp[pid].r, *v = pp[pid].v;
    enum {X, Y, Z};
    float sdf0 = k_sdf::sdf(r[X], r[Y], r[Z]);
    if (sdf0 >= 0)
      bounce0(sdf0, &r[X], &r[Y], &r[Z], &v[X], &v[Y], &v[Z]);
  }
}  /* namespace k_bb */
