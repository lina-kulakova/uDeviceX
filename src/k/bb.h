namespace k_bb { /* bounce back */
    __device__ void handle_collision(float currsdf,
				     float  &x, float  &y, float  &z,
				     float &vx, float &vy, float &vz) {
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
    float xw = x + t*vx, yw = y + t*vy, zw = z + t*vz; /* wall */
    x += 2*t*vx; y += 2*t*vy; z += 2*t*vz; /* bouncing relatively to
					      wall */
    k_wvel::bounce_vel(xw, yw, zw, &vx, &vy, &vz);
    if (k_sdf::sdf(x, y, z) >= 0) {x = x0; y = y0; z = z0;}
  }

  __global__ void bounce(Particle *pp, int n) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    float *r = pp[pid].r, *v = pp[pid].v;
    enum {X, Y, Z};
    float mycheapsdf = k_sdf::cheap_sdf(r[X], r[Y], r[Z]);
    if (mycheapsdf >= -1.7320f * (XE/(float)XTE)) {
      float sdf0 = k_sdf::sdf(r[X], r[Y], r[Z]);
      if (sdf0 >= 0)
	handle_collision(sdf0, r[X], r[Y], r[Z], v[X], v[Y], v[Z]);
    }
  }
}  /* namespace k_bb */
