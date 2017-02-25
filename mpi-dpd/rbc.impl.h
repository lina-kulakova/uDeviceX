using namespace std;

namespace CudaRBC
{
struct Params
{
  float kbT, p, lmax, kp, mpow, Area0, totArea0, totVolume0,
        kd, ka, kv, gammaT, gammaC,  sinTheta0, cosTheta0, kb, l0;
  float sint0kb, cost0kb, kbToverp;
  int  nvertices, ntriangles;
};

struct Extent
{
  float xmin, ymin, zmin;
  float xmax, ymax, zmax;
};

static Params params;
static __constant__ Params devParams;

/* blocking, initializes params */
void setup(int& nvertices, Extent& host_extent);

int get_nvertices();

/* A * (x, 1) */
void initialize(float *device_xyzuvw, const float (*transform)[4]);

/* non-synchronizing */
void forces_nohost(int ncells, const float * const device_xyzuvw, float * const device_axayaz);

/*non-synchronizing, extent not initialized */
void extent_nohost(int ncells, const float * const xyzuvw, Extent * device_extent, int n = -1);

/* get me a pointer to YOUR plain array - no allocation on my side */
void get_triangle_indexing(int (*&host_triplets_ptr)[3], int& ntriangles);
};

#include <cstdio>
#include <map>
#include <iostream>
#include <vector>
#include <algorithm>

#include "helper_math.h"

using namespace std;

extern float RBCx0, RBCp, RBCka, RBCkb, RBCkd, RBCkv, RBCgammaC,
       RBCtotArea, RBCtotVolume, RBCscale;

namespace CudaRBC
{

    texture<float2, 1, cudaReadModeElementType> texVertices;
    texture<int, 1, cudaReadModeElementType> texAdjVert;
    texture<int, 1, cudaReadModeElementType> texAdjVert2;
    texture<int4,   cudaTextureType1D> texTriangles4;

    float* orig_xyzuvw;
    float* host_av;
    float* devtrs4;

    int *triplets;

    float *addfrc;

    __constant__ float A[4][4];

    int maxCells;

    void unitsSetup(float x0, float p, float ka, float kb, float kd, float kv,
            float gammaC, float totArea0, float totVolume0, float lunit, float tunit, int ndens,
            bool prn);

    void eat_until(FILE * f, string target)
    {
        while(!feof(f))
        {
            char buf[2048];
            fgets(buf, 2048, f);

            if (string(buf) == target)
            {
                fgets(buf, 2048, f);
                break;
            }
        }
    }

    vector<int> extract_neighbors(vector<int> adjVert, const int degreemax, const int v)
    {
        vector<int> myneighbors;
        for(int c = 0; c < degreemax; ++c)
        {
            const int val = adjVert[c + degreemax * v];
            if (val == -1)
                break;

            myneighbors.push_back(val);
        }

        return myneighbors;
    }

    void setup_support(const int * data, const int * data2, const int nentries)
    {
        texAdjVert.channelDesc = cudaCreateChannelDesc<int>();
        texAdjVert.filterMode = cudaFilterModePoint;
        texAdjVert.mipmapFilterMode = cudaFilterModePoint;
        texAdjVert.normalized = 0;

        size_t textureoffset;
        CC(cudaBindTexture(&textureoffset, &texAdjVert, data,
                    &texAdjVert.channelDesc, sizeof(int) * nentries));

        texAdjVert2.channelDesc = cudaCreateChannelDesc<int>();
        texAdjVert2.filterMode = cudaFilterModePoint;
        texAdjVert2.mipmapFilterMode = cudaFilterModePoint;
        texAdjVert2.normalized = 0;

        CC(cudaBindTexture(&textureoffset, &texAdjVert2, data2,
                    &texAdjVert.channelDesc, sizeof(int) * nentries));
    }

    struct Particle
    {
        float x[3], u[3];
    };

    template <int nvertices>
        __global__ __launch_bounds__(128, 12)
        void fall_kernel(const int nrbcs, float* const __restrict__ av, float * const acc);

    void setup(int& nvertices, Extent& host_extent)
    {
        const float scale = RBCscale;
        const bool report = false;

        FILE * f = fopen("rbc.dat", "r");
        if (!f)
        {
            printf("Error in cuda-rbc: data file not found!\n");
            exit(1);
        }

        eat_until(f, "Atoms\n");

        vector<Particle> particles;
        while(!feof(f))
        {
            Particle p = {0, 0, 0, 0, 0, 0};
            int dummy[3];

            const int retval = fscanf(f, "%d %d %d %e %e %e\n", dummy + 0, dummy + 1, dummy + 2,
                    p.x, p.x+1, p.x+2);

            p.x[0] *= scale;
            p.x[1] *= scale;
            p.x[2] *= scale;

            if (retval != 6)
                break;

            particles.push_back(p);
        }

        eat_until(f, "Angles\n");

        vector< int3 > triangles;


        while(!feof(f))
        {
            int dummy[2];
            int3 tri;
            const int retval = fscanf(f, "%d %d %d %d %d\n", dummy + 0, dummy + 1,
                    &tri.x, &tri.y, &tri.z);

            if (retval != 5)
                break;

            triangles.push_back(tri);
        }
        fclose(f);

        triplets = new int[3*triangles.size()];
        int* trs4 = new int[4*triangles.size()];

        for (int i=0; i<triangles.size(); i++)
        {
            int3 tri = triangles[i];
            triplets[3*i + 0] = tri.x;
            triplets[3*i + 1] = tri.y;
            triplets[3*i + 2] = tri.z;

            trs4[4*i + 0] = tri.x;
            trs4[4*i + 1] = tri.y;
            trs4[4*i + 2] = tri.z;
            trs4[4*i + 3] = 0;
        }

        nvertices = particles.size();
        vector< map<int, int> > adjacentPairs(nvertices);

        for(int i = 0; i < triangles.size(); ++i)
        {
            const int tri[3] = {triangles[i].x, triangles[i].y, triangles[i].z};

            for(int d = 0; d < 3; ++d)
            {
                adjacentPairs[tri[d]][tri[(d + 1) % 3]] = tri[(d + 2) % 3];
            }

        }

        vector<int> maxldeg;
        for(int i = 0; i < nvertices; ++i)
            maxldeg.push_back(adjacentPairs[i].size());

        const int degreemax = *max_element(maxldeg.begin(), maxldeg.end());

        vector<int> adjVert(nvertices * degreemax, -1);

        for(int v = 0; v < nvertices; ++v)
        {
            map<int, int> l = adjacentPairs[v];

            adjVert[0 + degreemax * v] = l.begin()->first;
            int last = adjVert[1 + degreemax * v] = l.begin()->second;

            for(int i = 2; i < l.size(); ++i)
            {
                int tmp = adjVert[i + degreemax * v] = l.find(last)->second;
                last = tmp;
            }
        }

        vector<int> adjVert2(degreemax * nvertices, -1);

        for(int v = 0; v < nvertices; ++v)
        {
            vector<int> myneighbors = extract_neighbors(adjVert, degreemax, v);

            for(int i = 0; i < myneighbors.size(); ++i)
            {
                vector<int> s1 = extract_neighbors(adjVert, degreemax, myneighbors[i]);
                sort(s1.begin(), s1.end());

                vector<int> s2 = extract_neighbors(adjVert, degreemax, myneighbors[(i + 1) % myneighbors.size()]);
                sort(s2.begin(), s2.end());

                vector<int> result(s1.size() + s2.size());

                const int nterms =  set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
                        result.begin()) - result.begin();

                const int myguy = result[0] == v;

                adjVert2[i + degreemax * v] = result[myguy];
            }
        }

        params.nvertices = nvertices;
        params.ntriangles = triangles.size();

        // Find stretching points
        float stretchingForce = 0;
        vector< pair<float, int> > tmp(nvertices);
        for (int i=0; i<nvertices; i++)
        {
            tmp[i].first = particles[i].x[0];
            tmp[i].second = i;
        }
        sort(tmp.begin(), tmp.end());

        float* hAddfrc = new float[nvertices];
        memset(hAddfrc, 0, nvertices*sizeof(float));
        const int strVerts = 3; // 10
        for (int i=0; i<strVerts; i++)
        {
            hAddfrc[tmp[i].second] = -stretchingForce / strVerts;
            hAddfrc[tmp[nvertices - 1 - i].second] = +stretchingForce / strVerts;
        }

        CC( cudaMalloc(&addfrc, nvertices*sizeof(float)) );
        CC( cudaMemcpy(addfrc, hAddfrc, nvertices*sizeof(float), cudaMemcpyHostToDevice) );

        float* xyzuvw_host = new float[6*nvertices * sizeof(float)];
        for (int i=0; i<nvertices; i++)
        {
            xyzuvw_host[6*i+0] = particles[i].x[0];
            xyzuvw_host[6*i+1] = particles[i].x[1];
            xyzuvw_host[6*i+2] = particles[i].x[2];
            xyzuvw_host[6*i+3] = 0;
            xyzuvw_host[6*i+4] = 0;
            xyzuvw_host[6*i+5] = 0;
        }

        CC( cudaMalloc(&orig_xyzuvw, nvertices * 6 * sizeof(float)) );
        CC( cudaMemcpy(orig_xyzuvw, xyzuvw_host, nvertices * 6 * sizeof(float), cudaMemcpyHostToDevice) );
        delete[] xyzuvw_host;

        CC( cudaMalloc(&devtrs4, params.ntriangles * 4 * sizeof(int)) );
        CC( cudaMemcpy(devtrs4, trs4, params.ntriangles * 4 * sizeof(int), cudaMemcpyHostToDevice) );
        delete[] trs4;

        const int nentries = adjVert.size();

        int * ptr, * ptr2;
        CC(cudaMalloc(&ptr, sizeof(int) * nentries));
        CC(cudaMemcpy(ptr, &adjVert.front(), sizeof(int) * nentries, cudaMemcpyHostToDevice));

        CC(cudaMalloc(&ptr2, sizeof(int) * nentries));
        CC(cudaMemcpy(ptr2, &adjVert2.front(), sizeof(int) * nentries, cudaMemcpyHostToDevice));

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
        CC( cudaBindTexture(&textureoffset, &texTriangles4, devtrs4, &texTriangles4.channelDesc, params.ntriangles * 4 * sizeof(int)) );

        maxCells = 0;
        CC( cudaMalloc(&host_av, 1 * 2 * sizeof(float)) );

        unitsSetup(RBCx0, RBCp, RBCka, RBCkb, RBCkd, RBCkv, RBCgammaC,
                RBCtotArea, RBCtotVolume, 1e-6, -1 /* not used */, -1 /* not used */, report);
        CC( cudaFuncSetCacheConfig(fall_kernel<498>, cudaFuncCachePreferL1) );
    }

    void unitsSetup(float x0, float p, float ka, float kb, float kd, float kv,
            float gammaC, float totArea0, float totVolume0, float lunit, float tunit, int ndens,
            bool prn)
    {
        const float lrbc = 1.000000e-06;
        float ll =  (lunit/lrbc) / RBCscale;

        float kBT2D3D = 1;
        float phi = 6.97 / 180.0*M_PI; /* theta_0 */

        params.sinTheta0	= sin(phi);
        params.cosTheta0	= cos(phi);
        params.kbT		    = 0.1 * kBT2D3D;
        params.mpow			= 2; /* WLC-POW */

        /* units conversion: Fedosov -> uDeviceX */
        params.kv			= kv			* ll;
        params.gammaC		= gammaC		;
        params.ka			= ka			;
        params.kd			= kd			;
        params.p			= p				/ ll;
        params.totArea0		= totArea0		/ (ll*ll);
        params.kb			= kb			/ (ll*ll);
        params.kbT			= params.kbT	/ (ll*ll);
        params.totVolume0	= totVolume0	/ (ll*ll*ll);

        // derived parameters
        params.Area0		= params.totArea0 / (2.0*params.nvertices - 4.);
        params.l0			= sqrt(params.Area0 * 4.0/sqrt(3.0));
        params.lmax			= params.l0 / x0;
        params.gammaT		= 3.0 * params.gammaC;
        params.kbToverp		= params.kbT / params.p;
        params.sint0kb		= params.sinTheta0 * params.kb;
        params.cost0kb		= params.cosTheta0 * params.kb;
        params.kp			= ( params.kbT * x0 * (4*x0*x0-9*x0+6) * params.l0*params.l0 ) / ( 4*params.p*(x0-1)*(x0-1) );

        /* to simplify further computations */
        params.ka			= params.ka / params.totArea0;
        params.kv			= params.kv / (6 * params.totVolume0);

        CC( cudaMemcpyToSymbol  (devParams, &params, sizeof(Params)) );

        if (prn)
        {
            printf("\n************* Parameters setup *************\n");
            printf("Started with <RBC space (DPD space)>:\n");
            printf("    DPD unit of time:  %e\n",   tunit);
            printf("    DPD unit of length:  %e\n\n", lunit);
            printf("\t l0      %12.5f\n",           params.l0);
            printf("\t p       %12.5f  (%12.5f)\n", p,      params.p);
            printf("\t kb      %12.5f  (%12.5f)\n", kb,     params.kb);
            printf("\t ka      %12.5f  (%12.5f)\n", ka,     params.ka);
            printf("\t kv      %12.5f  (%12.5f)\n", kv,     params.kv);
            printf("\t gammaC  %12.5f  (%12.5f)\n\n", gammaC, params.gammaC);

            printf("\t kbT     %12e in dpd\n", params.kbT);
            printf("\t area    %12.5f  (%12.5f)\n", totArea0,  params.totArea0);
            printf("\t volume  %12.5f  (%12.5f)\n", totVolume0, params.totVolume0);
            printf("************* **************** *************\n\n");
        }
    }

    int get_nvertices()
    {
        return params.nvertices;
    }

    Params& get_params()
    {
        return params;
    }

    __global__ void transformKernel(float* xyzuvw, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        float x = xyzuvw[6*i + 0];
        float y = xyzuvw[6*i + 1];
        float z = xyzuvw[6*i + 2];

        xyzuvw[6*i + 0] = A[0][0]*x + A[0][1]*y + A[0][2]*z + A[0][3];
        xyzuvw[6*i + 1] = A[1][0]*x + A[1][1]*y + A[1][2]*z + A[1][3];
        xyzuvw[6*i + 2] = A[2][0]*x + A[2][1]*y + A[2][2]*z + A[2][3];
    }

    void initialize(float *device_xyzuvw, const float (*transform)[4])
    {
        const int threads = 128;
        const int blocks  = (params.nvertices + threads - 1) / threads;

        CC( cudaMemcpyToSymbol(A, transform, 16 * sizeof(float)) );
        CC( cudaMemcpy(device_xyzuvw, orig_xyzuvw, 6*params.nvertices * sizeof(float), cudaMemcpyDeviceToDevice) );
        transformKernel<<<blocks, threads>>>(device_xyzuvw, params.nvertices);
        CC( cudaPeekAtLastError() );
    }


    __device__ __forceinline__ float3 tex2vec(int id)
    {
        float2 tmp0 = tex1Dfetch(texVertices, id+0);
        float2 tmp1 = tex1Dfetch(texVertices, id+1);
        return make_float3(tmp0.x, tmp0.y, tmp1.x);
    }

    __device__ __forceinline__ float2 warpReduceSum(float2 val)
    {
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            val.x += __shfl_down(val.x, offset);
            val.y += __shfl_down(val.y, offset);
        }
        return val;
    }

    __global__ void areaAndVolumeKernel(float* totA_V)
    {
        float2 a_v = make_float2(0.0f, 0.0f);
        const int cid = blockIdx.y;

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < devParams.ntriangles; i += blockDim.x * gridDim.x)
        {
            int4 ids = tex1Dfetch(texTriangles4, i);

            float3 v0( tex2vec(3*(ids.x+cid*devParams.nvertices)) );
            float3 v1( tex2vec(3*(ids.y+cid*devParams.nvertices)) );
            float3 v2( tex2vec(3*(ids.z+cid*devParams.nvertices)) );

            a_v.x += 0.5f * length(cross(v1 - v0, v2 - v0));
            a_v.y += 0.1666666667f * (- v0.z*v1.y*v2.x + v0.z*v1.x*v2.y + v0.y*v1.z*v2.x
                    - v0.x*v1.z*v2.y - v0.y*v1.x*v2.z + v0.x*v1.y*v2.z);
        }

        a_v = warpReduceSum(a_v);
        if ((threadIdx.x & (warpSize - 1)) == 0)
        {
            atomicAdd(&totA_V[2*cid+0], a_v.x);
            atomicAdd(&totA_V[2*cid+1], a_v.y);
        }
    }

    // **************************************************************************************************

    __device__ __forceinline__ float3 _fangle(const float3 v1, const float3 v2, const float3 v3,
            const float area, const float volume)
    {
        const float3 x21 = v2 - v1;
        const float3 x32 = v3 - v2;
        const float3 x31 = v3 - v1;

        const float3 normal = cross(x21, x31);

        const float Ak = 0.5 * sqrtf(dot(normal, normal));
        const float A0 = devParams.Area0;
        const float n_2 = 1.0 / Ak;
        const float coefArea = -0.25f * (devParams.ka * (area - devParams.totArea0) * n_2)
            - devParams.kd*(Ak-A0)/(4.*A0*Ak);

        const float coeffVol = devParams.kv * (volume - devParams.totVolume0);
        const float3 addFArea = coefArea * cross(normal, x32);
        const float3 addFVolume = coeffVol * cross(v3, v2);

        float r = length(v2 - v1);
        r = r < 0.0001f ? 0.0001f : r;
        const float xx = r/devParams.lmax;
        const float IbforceI_wcl =
            devParams.kbToverp * ( 0.25f/((1.0f-xx)*(1.0f-xx)) - 0.25f + xx ) / r;
        const float kp    = devParams.kp;
        const float mpow  = devParams.mpow;
        const float IbforceI_pow = - kp / powf(r, mpow)                          / r;

        return addFArea + addFVolume + (IbforceI_wcl + IbforceI_pow ) * x21;
    }

    __device__ __forceinline__ float3 _fvisc(const float3 v1, const float3 v2, const float3 u1, const float3 u2)
    {
        const float3 du = u2 - u1;
        const float3 dr = v1 - v2;

        return du*devParams.gammaT + dr * devParams.gammaC*dot(du, dr) / dot(dr, dr);
    }

    template <int nvertices>
        __device__
        float3 _fangle_device(const float2 tmp0, const float2 tmp1, float* av)
        {
            const int degreemax = 7;
            const int pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;
            const int lid = pid % nvertices;
            const int idrbc = pid / nvertices;
            const int offset = idrbc * nvertices * 3;
            const int neighid = (threadIdx.x + blockDim.x * blockIdx.x) % degreemax;

            const float2 tmp2 = tex1Dfetch(texVertices, pid * 3 + 2);
            const float3 v1 = make_float3(tmp0.x, tmp0.y, tmp1.x);
            const float3 u1 = make_float3(tmp1.y, tmp2.x, tmp2.y);

            const int idv2 = tex1Dfetch(texAdjVert, neighid + degreemax * lid);
            bool valid = idv2 != -1;

            int idv3 = tex1Dfetch(texAdjVert, ((neighid + 1) % degreemax) + degreemax * lid);

            if (idv3 == -1 && valid)
                idv3 = tex1Dfetch(texAdjVert, 0 + degreemax * lid);

            if (valid)
            {
                const float2 tmp0 = tex1Dfetch(texVertices, offset + idv2 * 3 + 0);
                const float2 tmp1 = tex1Dfetch(texVertices, offset + idv2 * 3 + 1);
                const float2 tmp2 = tex1Dfetch(texVertices, offset + idv2 * 3 + 2);
                const float2 tmp3 = tex1Dfetch(texVertices, offset + idv3 * 3 + 0);
                const float2 tmp4 = tex1Dfetch(texVertices, offset + idv3 * 3 + 1);

                const float3 v2 = make_float3(tmp0.x, tmp0.y, tmp1.x);
                const float3 u2 = make_float3(tmp1.y, tmp2.x, tmp2.y);
                const float3 v3 = make_float3(tmp3.x, tmp3.y, tmp4.x);

                float3 f = _fangle(v1, v2, v3, av[2*idrbc], av[2*idrbc+1]);
                f += _fvisc(v1, v2, u1, u2);
                return f;
            }
            return make_float3(-1.0e10f, -1.0e10f, -1.0e10f);
        }

    //======================================
    //======================================

    template<int update>
        __device__  __forceinline__  float3 _fdihedral(float3 v1, float3 v2, float3 v3, float3 v4)
        {
            const float3 ksi   = cross(v1 - v2, v1 - v3);
            const float3 dzeta = cross(v3 - v4, v2 - v4);

            const float overIksiI   = rsqrtf(dot(ksi, ksi));
            const float overIdzetaI = rsqrtf(dot(dzeta, dzeta));

            const float cosTheta = dot(ksi, dzeta) * overIksiI*overIdzetaI;
            const float IsinThetaI2 = 1.0f - cosTheta*cosTheta;
            const float sinTheta_1 = copysignf( rsqrtf(max(IsinThetaI2, 1.0e-6f)), dot(ksi - dzeta, v4 - v1) );  // ">" because the normals look inside
            const float beta = devParams.cost0kb - cosTheta * devParams.sint0kb * sinTheta_1;

            float b11 = -beta * cosTheta * overIksiI*overIksiI;
            float b12 = beta * overIksiI*overIdzetaI;
            float b22 = -beta * cosTheta * overIdzetaI*overIdzetaI;

            if (update == 1)
                return cross(ksi, v3 - v2)*b11 + cross(dzeta, v3 - v2)*b12;
            else if (update == 2)
                return cross(ksi, v1 - v3)*b11 + ( cross(ksi, v3 - v4) + cross(dzeta, v1 - v3) )*b12 + cross(dzeta, v3 - v4)*b22;
            else return make_float3(0, 0, 0);
        }


    template <int nvertices>
        __device__
        float3 _fdihedral_device(const float2 tmp0, const float2 tmp1)
        {
            const int degreemax = 7;
            const int pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;
            const int lid = pid % nvertices;
            const int offset = (pid / nvertices) * nvertices * 3;
            const int neighid = (threadIdx.x + blockDim.x * blockIdx.x) % degreemax;

            const float3 v0 = make_float3(tmp0.x, tmp0.y, tmp1.x);

	    /*
                   v4
                 /   \
               v1 --> v2 --> v3
                 \   /
                   V
                   v0

             dihedrals: 0124, 0123
	    */

            int idv1, idv2, idv3, idv4;
            idv1 = tex1Dfetch(texAdjVert, neighid + degreemax * lid);
            const bool valid = idv1 != -1;

            idv2 = tex1Dfetch(texAdjVert, ((neighid + 1) % degreemax) +  degreemax * lid);

            if (idv2 == -1 && valid)
            {
                idv2 = tex1Dfetch(texAdjVert, 0 + degreemax * lid);
                idv3 = tex1Dfetch(texAdjVert, 1 + degreemax * lid);
            }
            else
            {
                idv3 = tex1Dfetch(texAdjVert, ((neighid + 2) % degreemax) +  degreemax * lid);
                if (idv3 == -1 && valid)
                    idv3 = tex1Dfetch(texAdjVert, 0 + degreemax * lid);
            }

            idv4 = tex1Dfetch(texAdjVert2, neighid + degreemax * lid);

            if (valid)
            {
                const float2 tmp0 = tex1Dfetch(texVertices, offset + idv1 * 3 + 0);
                const float2 tmp1 = tex1Dfetch(texVertices, offset + idv1 * 3 + 1);
                const float2 tmp2 = tex1Dfetch(texVertices, offset + idv2 * 3 + 0);
                const float2 tmp3 = tex1Dfetch(texVertices, offset + idv2 * 3 + 1);
                const float2 tmp4 = tex1Dfetch(texVertices, offset + idv3 * 3 + 0);
                const float2 tmp5 = tex1Dfetch(texVertices, offset + idv3 * 3 + 1);
                const float2 tmp6 = tex1Dfetch(texVertices, offset + idv4 * 3 + 0);
                const float2 tmp7 = tex1Dfetch(texVertices, offset + idv4 * 3 + 1);

                const float3 v1 = make_float3(tmp0.x, tmp0.y, tmp1.x);
                const float3 v2 = make_float3(tmp2.x, tmp2.y, tmp3.x);
                const float3 v3 = make_float3(tmp4.x, tmp4.y, tmp5.x);
                const float3 v4 = make_float3(tmp6.x, tmp6.y, tmp7.x);

                return _fdihedral<1>(v0, v2, v1, v4) + _fdihedral<2>(v1, v0, v2, v3);
            }
            return make_float3(-1.0e10f, -1.0e10f, -1.0e10f);
        }

    template <int nvertices>
        __global__ __launch_bounds__(128, 12)
        void fall_kernel(const int nrbcs, float* const __restrict__ av, float * const acc)
        {
            const int degreemax = 7;
            const int pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;

            if (pid < nrbcs * nvertices)
            {
                const float2 tmp0 = tex1Dfetch(texVertices, pid * 3 + 0);
                const float2 tmp1 = tex1Dfetch(texVertices, pid * 3 + 1);

                float3 f = _fangle_device<nvertices>(tmp0, tmp1, av);
                f += _fdihedral_device<nvertices>(tmp0, tmp1);

                if (f.x > -1.0e9f)
                {
                    atomicAdd(&acc[3*pid+0], f.x);
                    atomicAdd(&acc[3*pid+1], f.y);
                    atomicAdd(&acc[3*pid+2], f.z);
                }
            }
        }

    //

    __global__ void addKernel(float* axayaz, float* const __restrict__ addfrc, int n)
    {
        uint pid = threadIdx.x + blockIdx.x * blockDim.x;
        if (pid < n)
            axayaz[3*pid + 0] += addfrc[pid];

    }

    // **************************************************************************************************

    void forces_nohost(int ncells, const float * const device_xyzuvw, float * const device_axayaz)
    {
        if (ncells == 0) return;

        if (ncells > maxCells)
        {
            maxCells = 2*ncells;
            CC( cudaFree(host_av) );
            CC( cudaMalloc(&host_av, maxCells * 2 * sizeof(float)) );
        }

        size_t textureoffset;
        CC( cudaBindTexture(&textureoffset, &texVertices, (float2 *)device_xyzuvw,
                    &texVertices.channelDesc, ncells * params.nvertices * sizeof(float) * 6) );

        dim3 avThreads(256, 1);
        dim3 avBlocks( 1, ncells );

        CC( cudaMemsetAsync(host_av, 0, ncells * 2 * sizeof(float)) );
        areaAndVolumeKernel<<<avBlocks, avThreads, 0>>>(host_av);
        CC( cudaPeekAtLastError() );

        int threads = 128;
        int blocks  = (ncells*params.nvertices*7 + threads-1) / threads;

        fall_kernel<498><<<blocks, threads, 0>>>(ncells, host_av, device_axayaz);
        addKernel<<<(params.nvertices + 127) / 128, 128, 0>>>(device_axayaz, addfrc, params.nvertices);
    }

    void get_triangle_indexing(int (*&host_triplets_ptr)[3], int& ntriangles)
    {
        host_triplets_ptr = (int(*)[3])triplets;
        ntriangles = params.ntriangles;
    }

    float* get_orig_xyzuvw()
    {
        return orig_xyzuvw;
    }

    void extent_nohost(int ncells, const float * const xyzuvw, Extent * device_extent, int n)
    { }


}
