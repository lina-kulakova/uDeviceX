extern float RBCx0, RBCp, RBCka, RBCkb, RBCkd, RBCkv, RBCgammaC, RBCtotArea,
    RBCtotVolume, RBCscale;

namespace CudaRBC {

void unitsSetup(float x0, float p, float ka, float kb, float kd, float kv,
                float gammaC, float totArea0, float totVolume0, float lunit,
                float tunit, int ndens, bool prn) {
  const float lrbc = 1.000000e-06;
  float ll = (lunit / lrbc) / RBCscale;

  float kBT2D3D = 1;
  float phi = 6.97 / 180.0 * M_PI; /* theta_0 */

  params.sinTheta0 = sin(phi);
  params.cosTheta0 = cos(phi);
  params.kbT = 0.1 * kBT2D3D;
  params.mpow = 2; /* WLC-POW */

  /* units conversion: Fedosov -> uDeviceX */
  params.kv = kv * ll;
  params.gammaC = gammaC;
  params.ka = ka;
  params.kd = kd;
  params.p = p / ll;
  params.totArea0 = totArea0 / (ll * ll);
  params.kb = kb / (ll * ll);
  params.kbT = params.kbT / (ll * ll);
  params.totVolume0 = totVolume0 / (ll * ll * ll);

  // derived parameters
  params.Area0 = params.totArea0 / (2.0 * params.nvertices - 4.);
  params.l0 = sqrt(params.Area0 * 4.0 / sqrt(3.0));
  params.lmax = params.l0 / x0;
  params.gammaT = 3.0 * params.gammaC;
  params.kbToverp = params.kbT / params.p;
  params.sint0kb = params.sinTheta0 * params.kb;
  params.cost0kb = params.cosTheta0 * params.kb;
  params.kp =
      (params.kbT * x0 * (4 * x0 * x0 - 9 * x0 + 6) * params.l0 * params.l0) /
      (4 * params.p * (x0 - 1) * (x0 - 1));

  /* to simplify further computations */
  params.ka = params.ka / params.totArea0;
  params.kv = params.kv / (6 * params.totVolume0);

  CC(cudaMemcpyToSymbol(devParams, &params, sizeof(Params)));
}

void eat_until(FILE *f, std::string target) {
  while (!feof(f)) {
    char buf[2048];
    fgets(buf, 2048, f);

    if (std::string(buf) == target) {
      fgets(buf, 2048, f);
      break;
    }
  }
}

std::vector<int> extract_neighbors(std::vector<int> adjVert,
                                   const int degreemax, const int v) {
  std::vector<int> myneighbors;
  for (int c = 0; c < degreemax; ++c) {
    const int val = adjVert[c + degreemax * v];
    if (val == -1) break;

    myneighbors.push_back(val);
  }

  return myneighbors;
}

void setup_support(const int *data, const int *data2, const int nentries) {
  texAdjVert.channelDesc = cudaCreateChannelDesc<int>();
  texAdjVert.filterMode = cudaFilterModePoint;
  texAdjVert.mipmapFilterMode = cudaFilterModePoint;
  texAdjVert.normalized = 0;

  size_t textureoffset;
  CC(cudaBindTexture(&textureoffset, &texAdjVert, data, &texAdjVert.channelDesc,
                     sizeof(int) * nentries));

  texAdjVert2.channelDesc = cudaCreateChannelDesc<int>();
  texAdjVert2.filterMode = cudaFilterModePoint;
  texAdjVert2.mipmapFilterMode = cudaFilterModePoint;
  texAdjVert2.normalized = 0;

  CC(cudaBindTexture(&textureoffset, &texAdjVert2, data2,
                     &texAdjVert.channelDesc, sizeof(int) * nentries));
}

struct Particle {
  float x[3], u[3];
};

void setup(int &nvertices, Extent &host_extent) {
  const float scale = RBCscale;
  const bool report = false;

  FILE *f = fopen("rbc.dat", "r");
  if (!f) {
    printf("Error in cuda-rbc: data file not found!\n");
    exit(1);
  }

  eat_until(f, "Atoms\n");

  std::vector<Particle> particles;
  while (!feof(f)) {
    Particle p = {0, 0, 0, 0, 0, 0};
    int dummy[3];

    const int retval = fscanf(f, "%d %d %d %e %e %e\n", dummy + 0, dummy + 1,
                              dummy + 2, p.x, p.x + 1, p.x + 2);

    p.x[0] *= scale;
    p.x[1] *= scale;
    p.x[2] *= scale;

    if (retval != 6) break;

    particles.push_back(p);
  }

  eat_until(f, "Angles\n");

  std::vector<int3> triangles;

  while (!feof(f)) {
    int dummy[2];
    int3 tri;
    const int retval = fscanf(f, "%d %d %d %d %d\n", dummy + 0, dummy + 1,
                              &tri.x, &tri.y, &tri.z);

    if (retval != 5) break;

    triangles.push_back(tri);
  }
  fclose(f);

  triplets = new int[3 * triangles.size()];
  int *trs4 = new int[4 * triangles.size()];

  for (int i = 0; i < triangles.size(); i++) {
    int3 tri = triangles[i];
    triplets[3 * i + 0] = tri.x;
    triplets[3 * i + 1] = tri.y;
    triplets[3 * i + 2] = tri.z;

    trs4[4 * i + 0] = tri.x;
    trs4[4 * i + 1] = tri.y;
    trs4[4 * i + 2] = tri.z;
    trs4[4 * i + 3] = 0;
  }

  nvertices = particles.size();
  std::vector<std::map<int, int> > adjacentPairs(nvertices);

  for (int i = 0; i < triangles.size(); ++i) {
    const int tri[3] = {triangles[i].x, triangles[i].y, triangles[i].z};

    for (int d = 0; d < 3; ++d) {
      adjacentPairs[tri[d]][tri[(d + 1) % 3]] = tri[(d + 2) % 3];
    }
  }

  std::vector<int> maxldeg;
  for (int i = 0; i < nvertices; ++i)
    maxldeg.push_back(adjacentPairs[i].size());

  const int degreemax = *max_element(maxldeg.begin(), maxldeg.end());

  std::vector<int> adjVert(nvertices * degreemax, -1);

  for (int v = 0; v < nvertices; ++v) {
    std::map<int, int> l = adjacentPairs[v];

    adjVert[0 + degreemax * v] = l.begin()->first;
    int last = adjVert[1 + degreemax * v] = l.begin()->second;

    for (int i = 2; i < l.size(); ++i) {
      int tmp = adjVert[i + degreemax * v] = l.find(last)->second;
      last = tmp;
    }
  }

  std::vector<int> adjVert2(degreemax * nvertices, -1);

  for (int v = 0; v < nvertices; ++v) {
    std::vector<int> myneighbors = extract_neighbors(adjVert, degreemax, v);

    for (int i = 0; i < myneighbors.size(); ++i) {
      std::vector<int> s1 =
          extract_neighbors(adjVert, degreemax, myneighbors[i]);
      std::sort(s1.begin(), s1.end());

      std::vector<int> s2 = extract_neighbors(
          adjVert, degreemax, myneighbors[(i + 1) % myneighbors.size()]);
      std::sort(s2.begin(), s2.end());

      std::vector<int> result(s1.size() + s2.size());

      const int nterms = set_intersection(s1.begin(), s1.end(), s2.begin(),
                                          s2.end(), result.begin()) -
                         result.begin();

      const int myguy = result[0] == v;

      adjVert2[i + degreemax * v] = result[myguy];
    }
  }

  params.nvertices = nvertices;
  params.ntriangles = triangles.size();

  // Find stretching points
  float stretchingForce = 0;
  std::vector<std::pair<float, int> > tmp(nvertices);
  for (int i = 0; i < nvertices; i++) {
    tmp[i].first = particles[i].x[0];
    tmp[i].second = i;
  }
  sort(tmp.begin(), tmp.end());

  float *hAddfrc = new float[nvertices];
  memset(hAddfrc, 0, nvertices * sizeof(float));
  const int strVerts = 3; // 10
  for (int i = 0; i < strVerts; i++) {
    hAddfrc[tmp[i].second] = -stretchingForce / strVerts;
    hAddfrc[tmp[nvertices - 1 - i].second] = +stretchingForce / strVerts;
  }

  CC(cudaMalloc(&addfrc, nvertices * sizeof(float)));
  CC(cudaMemcpy(addfrc, hAddfrc, nvertices * sizeof(float),
                cudaMemcpyHostToDevice));

  float *xyzuvw_host = new float[6 * nvertices * sizeof(float)];
  for (int i = 0; i < nvertices; i++) {
    xyzuvw_host[6 * i + 0] = particles[i].x[0];
    xyzuvw_host[6 * i + 1] = particles[i].x[1];
    xyzuvw_host[6 * i + 2] = particles[i].x[2];
    xyzuvw_host[6 * i + 3] = 0;
    xyzuvw_host[6 * i + 4] = 0;
    xyzuvw_host[6 * i + 5] = 0;
  }

  CC(cudaMalloc(&orig_xyzuvw, nvertices * 6 * sizeof(float)));
  CC(cudaMemcpy(orig_xyzuvw, xyzuvw_host, nvertices * 6 * sizeof(float),
                cudaMemcpyHostToDevice));
  delete[] xyzuvw_host;

  CC(cudaMalloc(&devtrs4, params.ntriangles * 4 * sizeof(int)));
  CC(cudaMemcpy(devtrs4, trs4, params.ntriangles * 4 * sizeof(int),
                cudaMemcpyHostToDevice));
  delete[] trs4;

  const int nentries = adjVert.size();

  int *ptr, *ptr2;
  CC(cudaMalloc(&ptr, sizeof(int) * nentries));
  CC(cudaMemcpy(ptr, &adjVert.front(), sizeof(int) * nentries,
                cudaMemcpyHostToDevice));

  CC(cudaMalloc(&ptr2, sizeof(int) * nentries));
  CC(cudaMemcpy(ptr2, &adjVert2.front(), sizeof(int) * nentries,
                cudaMemcpyHostToDevice));

  setup_support(ptr, ptr2, nentries);

  texTriangles4.channelDesc = cudaCreateChannelDesc<int4>();
  texTriangles4.filterMode = cudaFilterModePoint;
  texTriangles4.mipmapFilterMode = cudaFilterModePoint;
  texTriangles4.normalized = 0;

  texVertices.channelDesc = cudaCreateChannelDesc<float2>();
  texVertices.filterMode = cudaFilterModePoint;
  texVertices.mipmapFilterMode = cudaFilterModePoint;
  texVertices.normalized = 0;

  size_t textureoffset;
  CC(cudaBindTexture(&textureoffset, &texTriangles4, devtrs4,
                     &texTriangles4.channelDesc,
                     params.ntriangles * 4 * sizeof(int)));

  maxCells = 0;
  CC(cudaMalloc(&host_av, 1 * 2 * sizeof(float)));

  unitsSetup(RBCx0, RBCp, RBCka, RBCkb, RBCkd, RBCkv, RBCgammaC, RBCtotArea,
             RBCtotVolume, 1e-6, -1 /* not used */, -1 /* not used */, report);
  CC(cudaFuncSetCacheConfig(fall_kernel<498>, cudaFuncCachePreferL1));
}

int get_nvertices() { return params.nvertices; }

Params &get_params() { return params; }

__global__ void transformKernel(float *xyzuvw, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float x = xyzuvw[6 * i + 0];
  float y = xyzuvw[6 * i + 1];
  float z = xyzuvw[6 * i + 2];

  xyzuvw[6 * i + 0] = A[0][0] * x + A[0][1] * y + A[0][2] * z + A[0][3];
  xyzuvw[6 * i + 1] = A[1][0] * x + A[1][1] * y + A[1][2] * z + A[1][3];
  xyzuvw[6 * i + 2] = A[2][0] * x + A[2][1] * y + A[2][2] * z + A[2][3];
}

void initialize(float *device_xyzuvw, const float (*transform)[4]) {
  const int threads = 128;
  const int blocks = (params.nvertices + threads - 1) / threads;

  CC(cudaMemcpyToSymbol(A, transform, 16 * sizeof(float)));
  CC(cudaMemcpy(device_xyzuvw, orig_xyzuvw,
                6 * params.nvertices * sizeof(float),
                cudaMemcpyDeviceToDevice));
  transformKernel<<<blocks, threads>>>(device_xyzuvw, params.nvertices);
  CC(cudaPeekAtLastError());
}

__device__ __forceinline__ float3 tex2vec(int id) {
  float2 tmp0 = tex1Dfetch(texVertices, id + 0);
  float2 tmp1 = tex1Dfetch(texVertices, id + 1);
  return make_float3(tmp0.x, tmp0.y, tmp1.x);
}

__device__ __forceinline__ float2 warpReduceSum(float2 val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val.x += __shfl_down(val.x, offset);
    val.y += __shfl_down(val.y, offset);
  }
  return val;
}

__global__ void areaAndVolumeKernel(float *totA_V) {
  float2 a_v = make_float2(0.0f, 0.0f);
  const int cid = blockIdx.y;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < devParams.ntriangles;
       i += blockDim.x * gridDim.x) {
    int4 ids = tex1Dfetch(texTriangles4, i);

    float3 v0(tex2vec(3 * (ids.x + cid * devParams.nvertices)));
    float3 v1(tex2vec(3 * (ids.y + cid * devParams.nvertices)));
    float3 v2(tex2vec(3 * (ids.z + cid * devParams.nvertices)));

    a_v.x += 0.5f * length(cross(v1 - v0, v2 - v0));
    a_v.y += 0.1666666667f *
             (-v0.z * v1.y * v2.x + v0.z * v1.x * v2.y + v0.y * v1.z * v2.x -
              v0.x * v1.z * v2.y - v0.y * v1.x * v2.z + v0.x * v1.y * v2.z);
  }

  a_v = warpReduceSum(a_v);
  if ((threadIdx.x & (warpSize - 1)) == 0) {
    atomicAdd(&totA_V[2 * cid + 0], a_v.x);
    atomicAdd(&totA_V[2 * cid + 1], a_v.y);
  }
}

void forces_nohost(int ncells, const float *const device_xyzuvw,
                   float *const device_axayaz) {
  if (ncells == 0) return;

  if (ncells > maxCells) {
    maxCells = 2 * ncells;
    CC(cudaFree(host_av));
    CC(cudaMalloc(&host_av, maxCells * 2 * sizeof(float)));
  }

  size_t textureoffset;
  CC(cudaBindTexture(&textureoffset, &texVertices, (float2 *)device_xyzuvw,
                     &texVertices.channelDesc,
                     ncells * params.nvertices * sizeof(float) * 6));

  dim3 avThreads(256, 1);
  dim3 avBlocks(1, ncells);

  CC(cudaMemsetAsync(host_av, 0, ncells * 2 * sizeof(float)));
  areaAndVolumeKernel<<<avBlocks, avThreads, 0>>>(host_av);
  CC(cudaPeekAtLastError());

  int threads = 128;
  int blocks = (ncells * params.nvertices * 7 + threads - 1) / threads;

  fall_kernel<498><<<blocks, threads, 0>>>(ncells, host_av, device_axayaz);
  addKernel<<<(params.nvertices + 127) / 128, 128, 0>>>(device_axayaz, addfrc,
                                                        params.nvertices);
}

void get_triangle_indexing(int (*&host_triplets_ptr)[3], int &ntriangles) {
  host_triplets_ptr = (int(*)[3])triplets;
  ntriangles = params.ntriangles;
}

float *get_orig_xyzuvw() { return orig_xyzuvw; }

void extent_nohost(int ncells, const float *const xyzuvw, Extent *device_extent,
                   int n) {}
}
