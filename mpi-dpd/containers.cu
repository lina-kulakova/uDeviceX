
/*
 *  containers.cu
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-05.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <sys/stat.h>

#include <rbc-cuda.h>

#include "common.h"
#include "containers.h"
#include "io.h"
#include "last_bit_float.h"
#include "dpd-forces.h"

int (*CollectionRBC::indices)[3] = NULL, CollectionRBC::ntriangles = -1, CollectionRBC::nvertices = -1;


namespace ParticleKernels
{
    __global__ void update_stage1(Particle * p, Acceleration * a, int n, float dt,
            const float _driving_acceleration, const float threshold, const bool doublePoiseuille, const bool check = true)
    {
        const int pid = threadIdx.x + blockDim.x * blockIdx.x;

        if (pid >= n)
            return;

        last_bit_float::Preserver up0(p[pid].u[0]);

        float driving_acceleration = _driving_acceleration;

        if (doublePoiseuille && p[pid].x[1] <= threshold)
            driving_acceleration *= -1;

        for(int c = 0; c < 3; ++c)
            p[pid].u[c] += (a[pid].a[c] + (c == 0 ? driving_acceleration : 0)) * dt * 0.5;

        for(int c = 0; c < 3; ++c)
            p[pid].x[c] += p[pid].u[c] * dt;

    }

    __global__ void update_stage2_and_1(float2 * const _pdata, const float * const _adata,
            const int nparticles, const float dt, const float _driving_acceleration, const float threshold,
            const bool doublePoiseuille)
    {

#if !defined(__CUDA_ARCH__)
#warning __CUDA_ARCH__ not defined! assuming 350
#define _ACCESS(x) __ldg(x)
#elif __CUDA_ARCH__ >= 350
#define _ACCESS(x) __ldg(x)
#else
#define _ACCESS(x) (*(x))
#endif

        const int warpid = threadIdx.x >> 5;
        const int base = 32 * (warpid + 4 * blockIdx.x);
        const int nsrc = min(32, nparticles - base);

        float2 * const pdata = _pdata + 3 * base;
        const float * const adata = _adata + 3 * base;

        int laneid;
        asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));

        const int nwords = 3 * nsrc;

        float2 s0, s1, s2;
        float ax, ay, az;

        if (laneid < nwords)
        {
            s0 = _ACCESS(pdata + laneid);
            ax = _ACCESS(adata + laneid);
        }

        if (laneid + 32 < nwords)
        {
            s1 = _ACCESS(pdata + laneid + 32);
            ay = _ACCESS(adata + laneid + 32);
        }

        if (laneid + 64 < nwords)
        {
            s2 = _ACCESS(pdata + laneid + 64);
            az = _ACCESS(adata + laneid + 64);
        }

        {
            const int srclane0 = (3 * laneid + 0) & 0x1f;
            const int srclane1 = (srclane0 + 1) & 0x1f;
            const int srclane2 = (srclane0 + 2) & 0x1f;

            const int start = laneid % 3;

            {
                const float t0 = __shfl(start == 0 ? s0.x : start == 1 ? s1.x : s2.x, srclane0);
                const float t1 = __shfl(start == 0 ? s2.x : start == 1 ? s0.x : s1.x, srclane1);
                const float t2 = __shfl(start == 0 ? s1.x : start == 1 ? s2.x : s0.x, srclane2);

                s0.x = t0;
                s1.x = t1;
                s2.x = t2;
            }

            {
                const float t0 = __shfl(start == 0 ? s0.y : start == 1 ? s1.y : s2.y, srclane0);
                const float t1 = __shfl(start == 0 ? s2.y : start == 1 ? s0.y : s1.y, srclane1);
                const float t2 = __shfl(start == 0 ? s1.y : start == 1 ? s2.y : s0.y, srclane2);

                s0.y = t0;
                s1.y = t1;
                s2.y = t2;
            }

            {
                const float t0 = __shfl(start == 0 ? ax : start == 1 ? ay : az, srclane0);
                const float t1 = __shfl(start == 0 ? az : start == 1 ? ax : ay, srclane1);
                const float t2 = __shfl(start == 0 ? ay : start == 1 ? az : ax, srclane2);

                ax = t0;
                ay = t1;
                az = t2;
            }
        }

        float driving_acceleration = _driving_acceleration;

        if (doublePoiseuille && s0.y <= threshold)
            driving_acceleration *= -1;

        s1.y += (ax + driving_acceleration) * dt;
        s2.x += ay * dt;
        s2.y += az * dt;

        s0.x += s1.y * dt;
        s0.y += s2.x * dt;
        s1.x += s2.y * dt;

        {
            const int srclane0 = (32 * ((laneid) % 3) + laneid) / 3;
            const int srclane1 = (32 * ((laneid + 1) % 3) + laneid) / 3;
            const int srclane2 = (32 * ((laneid + 2) % 3) + laneid) / 3;

            const int start = laneid % 3;

            {
                const float t0 = __shfl(s0.x, srclane0);
                const float t1 = __shfl(s2.x, srclane1);
                const float t2 = __shfl(s1.x, srclane2);

                s0.x = start == 0 ? t0 : start == 1 ? t2 : t1;
                s1.x = start == 0 ? t1 : start == 1 ? t0 : t2;
                s2.x = start == 0 ? t2 : start == 1 ? t1 : t0;
            }

            {
                const float t0 = __shfl(s0.y, srclane0);
                const float t1 = __shfl(s2.y, srclane1);
                const float t2 = __shfl(s1.y, srclane2);

                s0.y = start == 0 ? t0 : start == 1 ? t2 : t1;
                s1.y = start == 0 ? t1 : start == 1 ? t0 : t2;
                s2.y = start == 0 ? t2 : start == 1 ? t1 : t0;
            }

            {
                const float t0 = __shfl(ax, srclane0);
                const float t1 = __shfl(az, srclane1);
                const float t2 = __shfl(ay, srclane2);

                ax = start == 0 ? t0 : start == 1 ? t2 : t1;
                ay = start == 0 ? t1 : start == 1 ? t0 : t2;
                az = start == 0 ? t2 : start == 1 ? t1 : t0;
            }
        }

        if (laneid < nwords) {
            last_bit_float::Preserver up1(pdata[laneid].y);
            pdata[laneid] = s0;
        }

        if (laneid + 32 < nwords) {
            last_bit_float::Preserver up1(pdata[laneid + 32].y);
            pdata[laneid + 32] = s1;
        }

        if (laneid + 64 < nwords) {
            last_bit_float::Preserver up1(pdata[laneid + 64].y);
            pdata[laneid + 64] = s2;
        }
    }

    __global__ void clear_velocity(Particle * const p, const int n)
    {
        const int pid = threadIdx.x + blockDim.x * blockIdx.x;

        if (pid >= n)
            return;

        last_bit_float::Preserver up(p[pid].u[0]);
        for(int c = 0; c < 3; ++c)
            p[pid].u[c] = 0;
    }
}

void ParticleArray::update_stage1(const float driving_acceleration, cudaStream_t stream)
{
    if (size)
        ParticleKernels::update_stage1<<<(xyzuvw.size + 127) / 128, 128, 0, stream>>>(
                xyzuvw.data, axayaz.data, xyzuvw.size, dt, driving_acceleration, globalextent.y * 0.5 - origin.y, doublepoiseuille, false);
}

void  ParticleArray::update_stage2_and_1(const float driving_acceleration, cudaStream_t stream)
{
    if (size)
        ParticleKernels::update_stage2_and_1<<<(xyzuvw.size + 127) / 128, 128, 0, stream>>>
            ((float2 *)xyzuvw.data, (float *)axayaz.data, xyzuvw.size, dt, driving_acceleration, globalextent.y * 0.5 - origin.y, doublepoiseuille);
}

void ParticleArray::resize(int n)
{
    size = n;

    // YTANG: need the array to be 32-padded for locally transposed storage of acceleration
    if ( n % 32 ) {
        xyzuvw.preserve_resize( n - n % 32 + 32 );
        axayaz.preserve_resize( n - n % 32 + 32 );
    }
    xyzuvw.resize(n);
    axayaz.resize(n);

#ifndef NDEBUG
    CUDA_CHECK(cudaMemset(axayaz.data, 0, sizeof(Acceleration) * size));
#endif
}

void ParticleArray::preserve_resize(int n)
{
    int oldsize = size;
    size = n;

    xyzuvw.preserve_resize(n);
    axayaz.preserve_resize(n);

    if (size > oldsize)
        CUDA_CHECK(cudaMemset(axayaz.data + oldsize, 0, sizeof(Acceleration) * (size-oldsize)));
}

void ParticleArray::clear_velocity()
{
    if (size)
        ParticleKernels::clear_velocity<<<(xyzuvw.size + 127) / 128, 128 >>>(xyzuvw.data, xyzuvw.size);
}

void CollectionRBC::resize(const int count)
{
    ncells = count;

    ParticleArray::resize(count * get_nvertices());
}

void CollectionRBC::preserve_resize(const int count)
{
    ncells = count;

    ParticleArray::preserve_resize(count * get_nvertices());
}

struct TransformedExtent
{
    float com[3];
    float transform[4][4];
};

CollectionRBC::CollectionRBC(MPI_Comm cartcomm):
    cartcomm(cartcomm), ncells(0)
{
    MPI_CHECK(MPI_Comm_rank(cartcomm, &myrank));
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    CudaRBC::get_triangle_indexing(indices, ntriangles);
    CudaRBC::Extent extent;
    CudaRBC::setup(nvertices, extent);
}

void CollectionRBC::setup(const char * const path2ic)
{
    vector<TransformedExtent> allrbcs;

    if (myrank == 0)
    {
        //read transformed extent from file
        FILE * f = fopen(path2ic, "r");
        printf("READING FROM: <%s>\n", path2ic);
        bool isgood = true;

        while(isgood)
        {
            float tmp[19];
            for(int c = 0; c < 19; ++c)
            {
                int retval = fscanf(f, "%f", tmp + c);

                isgood &= retval == 1;
            }

            if (isgood)
            {
                TransformedExtent t;

                for(int c = 0; c < 3; ++c)
                    t.com[c] = tmp[c];

                int ctr = 3;
                for(int c = 0; c < 16; ++c, ++ctr)
                    t.transform[c / 4][c % 4] = tmp[ctr];

                allrbcs.push_back(t);
            }
        }

        fclose(f);
    }

    if (myrank == 0)
        printf("Instantiating %d CELLs from...<%s>\n", (int)allrbcs.size(), path2ic);

    int allrbcs_count = allrbcs.size();
    MPI_CHECK(MPI_Bcast(&allrbcs_count, 1, MPI_INT, 0, cartcomm));

    allrbcs.resize(allrbcs_count);

    const int nfloats_per_entry = sizeof(TransformedExtent) / sizeof(float);

    MPI_CHECK(MPI_Bcast(&allrbcs.front(), nfloats_per_entry * allrbcs_count, MPI_FLOAT, 0, cartcomm));

    vector<TransformedExtent> good;

    const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };

    for(vector<TransformedExtent>::iterator it = allrbcs.begin(); it != allrbcs.end(); ++it)
    {
        bool inside = true;

        for(int c = 0; c < 3; ++c)
            inside &= it->com[c] >= coords[c] * L[c] && it->com[c] < (coords[c] + 1) * L[c];

        if (inside)
        {
            for(int c = 0; c < 3; ++c)
                it->transform[c][3] -= (coords[c] + 0.5) * L[c];

            good.push_back(*it);
        }
    }

    resize(good.size());

    for(int i = 0; i < good.size(); ++i)
        _initialize((float *)(xyzuvw.data + get_nvertices() * i), good[i].transform);
}

void CollectionRBC::_initialize(float *device_xyzuvw, const float (*transform)[4])
{
    CudaRBC::initialize(device_xyzuvw, transform);
}

void CollectionRBC::remove(const int * const entries, const int nentries)
{
    std::vector<bool > marks(ncells, true);

    for(int i = 0; i < nentries; ++i)
        marks[entries[i]] = false;

    std::vector< int > survivors;
    for(int i = 0; i < ncells; ++i)
        if (marks[i])
            survivors.push_back(i);

    const int nsurvived = survivors.size();

    SimpleDeviceBuffer<Particle> survived(get_nvertices() * nsurvived);

    for(int i = 0; i < nsurvived; ++i)
        CUDA_CHECK(cudaMemcpy(survived.data + get_nvertices() * i, data() + get_nvertices() * survivors[i],
                    sizeof(Particle) * get_nvertices(), cudaMemcpyDeviceToDevice));

    resize(nsurvived);

    CUDA_CHECK(cudaMemcpy(xyzuvw.data, survived.data, sizeof(Particle) * survived.size, cudaMemcpyDeviceToDevice));
}

void CollectionRBC::_dump(const char * const path2xyz, const char * const format4ply,
        MPI_Comm comm, MPI_Comm cartcomm, const int ntriangles, const int ncells, const int nvertices, int (* const indices)[3],
        Particle * const p, const Acceleration * const a, const int n, const int iddatadump)
{
    int ctr = iddatadump;
    const bool firsttime = ctr == 0;

    //we fused VV stages so we need to recover the state before stage 1
    for(int i = 0; i < n; ++i) {
        last_bit_float::Preserver up(p[i].u[0]);
        for(int c = 0; c < 3; ++c)
        {
            p[i].x[c] -= dt * p[i].u[c];
            p[i].u[c] -= 0.5 * dt * a[i].a[c];
        }
    }


    char buf[200];
    sprintf(buf, format4ply, ctr);

    {
        int rank;
        MPI_CHECK(MPI_Comm_rank(comm, &rank));

        if(rank == 0)
            mkdir("ply", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }

    ply_dump(comm, cartcomm, buf, indices, ncells, ntriangles, p, nvertices, false);
}
