namespace k_sdf {
  texture<float, 3, cudaReadModeElementType> texSDF;
  __device__ void  r2q(float* r, /**/ float* q) {
    /* from subdomain to textrue coordinates */
    int L[3] = {XS, YS, ZS}, WM[3] = {XWM, YWM, ZWM}, \
			     TE[3] = {XTE, YTE, ZTE};
    for (int c = 0; c < 3; c++)
      q[c] = TE[c] * (r[c] + 0.5*L[c] + WM[c]) / (float)(L[c] + 2 * WM[c]) - 0.5;
  }

  __device__ float sdf(float x, float y, float z) {
    float tc[3], lmbd[3], q[3], r[3] = {x, y, z};
    r2q(r, /**/ q);
    for (int c = 0; c < 3; ++c) {
      float t = q[c];
      lmbd[c] = t - (int)t;
      tc[c] = (int)t + 0.5;
    }
#define tex0(ix, iy, iz) (tex3D(texSDF, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    float s000 = tex0(0, 0, 0), s001 = tex0(1, 0, 0), s010 = tex0(0, 1, 0);
    float s011 = tex0(1, 1, 0), s100 = tex0(0, 0, 1), s101 = tex0(1, 0, 1);
    float s110 = tex0(0, 1, 1), s111 = tex0(1, 1, 1);
#undef tex0

#define wavrg(A, B, p) A*(1-p) + p*B /* weighted average */
    float s00x = wavrg(s000, s001, lmbd[0]);
    float s01x = wavrg(s010, s011, lmbd[0]);
    float s10x = wavrg(s100, s101, lmbd[0]);
    float s11x = wavrg(s110, s111, lmbd[0]);

    float s0yx = wavrg(s00x, s01x, lmbd[1]);

    float s1yx = wavrg(s10x, s11x, lmbd[1]);
    float szyx = wavrg(s0yx, s1yx, lmbd[2]);
#undef wavrg
    return szyx;
  }

  __device__ float3 ugrad_sdf(float x, float y, float z) {
    int L[3] = {XS, YS, ZS}, WM[3] = {XWM, YWM, ZWM}, \
			     TE[3] = {XTE, YTE, ZTE};
    float fcts[3], q[3], r[3] = {x, y, z};
    int   tc[3];
    int c;
    r2q(r, /**/ q);
    for (c = 0; c < 3; ++c) tc[c] = (int)q[c];
    for (c = 0; c < 3; ++c) fcts[c] = TE[c]/(float)(2*WM[c] + L[c]);

#define tex0(ix, iy, iz) (tex3D(texSDF, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    float gx, gy, gz, sdf0 = tex0(0, 0, 0);
    gx = fcts[0] * (tex0(1, 0, 0) - sdf0);
    gy = fcts[1] * (tex0(0, 1, 0) - sdf0);
    gz = fcts[2] * (tex0(0, 0, 1) - sdf0);
#undef tex0
    return make_float3(gx, gy, gz);
  }

  __global__ void fill_keys(Particle *pp, int n, int *key) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    float sdf0, *r = pp[pid].r;
    enum {X, Y, Z};
    sdf0 = sdf(r[X], r[Y], r[Z]);

    key[pid] = \
      sdf0  > 2 ? W_DEEP :
      sdf0 >= 0 ? W_WALL :
		  W_BULK;
  }
} /* namespace k_sdf */
