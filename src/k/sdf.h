namespace k_sdf {
  texture<float, 3, cudaReadModeElementType> texSDF;

  __device__ float sdf(float x, float y, float z) {
    int L[3] = {XS, YS, ZS};
    int MARGIN[3] = {XWM, YWM, ZWM};
    int TEXSIZES[3] = {XTE, YTE, ZTE};

    float tc[3], lmbd[3], r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c) {
      float t =
	TEXSIZES[c] * (r[c] + L[c] / 2 + MARGIN[c]) / (L[c] + 2 * MARGIN[c]);

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

  /* within the rescaled texel width error */
  __device__ float cheap_sdf(float x, float y, float z)  {
    int L[3] = {XS, YS, ZS};
    int MARGIN[3] = {XWM, YWM, ZWM};
    int TEXSIZES[3] = {XTE, YTE, ZTE};

    float tc[3], r[3] = {x, y, z};;
    for (int c = 0; c < 3; ++c)
      tc[c] = 0.5001f + (int)(TEXSIZES[c] * (r[c] + L[c] / 2 + MARGIN[c]) /
			      (L[c] + 2 * MARGIN[c]));
#define tex0(ix, iy, iz) (tex3D(texSDF, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    return tex0(0, 0, 0);
#undef  tex0
  }

  __device__ float3 ugrad_sdf(float x, float y, float z) {
    int L[3] = {XS, YS, ZS};
    int MARGIN[3] = {XWM, YWM, ZWM};
    int TEXSIZES[3] = {XTE, YTE, ZTE};

    float tc[3], fcts[3], r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c)
      tc[c] = 0.5001f + (int)(TEXSIZES[c] * (r[c] + L[c] / 2 + MARGIN[c]) /
			      (L[c] + 2 * MARGIN[c]));
    for (int c = 0; c < 3; ++c) fcts[c] = TEXSIZES[c] / (2 * MARGIN[c] + L[c]);

#define tex0(ix, iy, iz) (tex3D(texSDF, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    float myval = tex0(0, 0, 0);
    float gx = fcts[0] * (tex0(1, 0, 0) - myval);
    float gy = fcts[1] * (tex0(0, 1, 0) - myval);
    float gz = fcts[2] * (tex0(0, 0, 1) - myval);
#undef tex0

    return make_float3(gx, gy, gz);
  }

  __device__ float3 grad_sdf(float x, float y, float z) {
    int L[3] = {XS, YS, ZS};
    int MARGIN[3] = {XWM, YWM, ZWM};
    int TEXSIZES[3] = {XTE, YTE, ZTE};

    float tc[3], r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c)
      tc[c] =
	TEXSIZES[c] * (r[c] + L[c] / 2 + MARGIN[c]) / (L[c] + 2 * MARGIN[c]);

    float gx, gy, gz;
#define tex0(ix, iy, iz) (tex3D(texSDF, tc[0] + ix, tc[1] + iy, tc[2] + iz))
    gx = tex0(1, 0, 0) - tex0(-1,  0,  0);
    gy = tex0(0, 1, 0) - tex0( 0, -1,  0);
    gz = tex0(0, 0, 1) - tex0( 0,  0, -1);
#undef tex0

    float ggmag = sqrt(gx*gx + gy*gy + gz*gz);

    if (ggmag > 1e-6) {
      gx /= ggmag; gy /= ggmag; gz /= ggmag;
    }
    return make_float3(gx, gy, gz);
  }

  __global__ void fill_keys(Particle *pp, int n, int *key) {
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    float sdf0, *r = pp[pid].r;
    sdf0 = sdf(r[0], r[1], r[2]);

    key[pid] = \
      sdf0  > 2 ? W_DEEP :
      sdf0 >= 0 ? W_WALL :
		  W_BULK;
  }
} /* namespace k_sdf */
