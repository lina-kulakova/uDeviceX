namespace k_bb { /* bounce back */
    __device__ void handle_collision(float currsdf,
				   float &x, float &y, float &z,
				   float &vx, float &vy, float &vz) {
    float x0 = x - vx*dt, y0 = y - vy*dt, z0 = z - vz*dt;
    if (k_sdf::sdf(x0, y0, z0) >= 0) { /* this is the worst case - 0 position
				   was bad already we need to search
				   and rescue the particle */
      float3 dsdf = k_sdf::grad_sdf(x, y, z); float sdf0 = currsdf;
      x -= sdf0 * dsdf.x; y -= sdf0 * dsdf.y; z -= sdf0 * dsdf.z;
      for (int l = 8; l >= 1; --l) {
	if (k_sdf::sdf(x, y, z) < 0) {
	  /* we are confused anyway! use particle position as wall
	     position */
	  k_wvel::bounce_vel(x, y, z, &vx, &vy, &vz); return;
	}
	float jump = 1.1f * sdf0 / (1 << l);
	x -= jump * dsdf.x; y -= jump * dsdf.y; z -= jump * dsdf.z;
      }
    }

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

  __global__ void bounce(float2 *const pp, int nparticles) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= nparticles) return;
    float2 data0 = pp[pid * 3];
    float2 data1 = pp[pid * 3 + 1];
    if (pid < nparticles) {
      float mycheapsdf = k_sdf::cheap_sdf(data0.x, data0.y, data1.x);

      if (mycheapsdf >=
	  -1.7320f * ((float)XSIZE_WALLCELLS / (float)XTEXTURESIZE)) {
	float currsdf = k_sdf::sdf(data0.x, data0.y, data1.x);

	float2 data2 = pp[pid * 3 + 2];

	float3 v0 = make_float3(data1.y, data2.x, data2.y);

	if (currsdf >= 0) {
	  handle_collision(currsdf, data0.x, data0.y, data1.x, data1.y, data2.x,
			   data2.y);

	  pp[3 * pid] = data0;
	  pp[3 * pid + 1] = data1;
	  pp[3 * pid + 2] = data2;
	}
      }
    }
  }
}  /* namespace k_bb */
