#include <../saru.cuh>

#include "rbc-interactions.h"

namespace KernelsRBC
{
    struct ParamsFSI
    {
	float aij, gamma, sigmaf;
    };

    __constant__ ParamsFSI params;
    
    texture<float2, cudaTextureType1D> texSolventParticles;
    texture<int, cudaTextureType1D> texCellsStart, texCellsCount;

    static bool firsttime = true;
    
    void setup(const Particle * const solvent, const int npsolvent, const int * const cellsstart, const int * const cellscount, const int L)
    {
	if (firsttime)
	{
	    texCellsStart.channelDesc = cudaCreateChannelDesc<int>();
	    texCellsStart.filterMode = cudaFilterModePoint;
	    texCellsStart.mipmapFilterMode = cudaFilterModePoint;
	    texCellsStart.normalized = 0;
    
	    texCellsCount.channelDesc = cudaCreateChannelDesc<int>();
	    texCellsCount.filterMode = cudaFilterModePoint;
	    texCellsCount.mipmapFilterMode = cudaFilterModePoint;
	    texCellsCount.normalized = 0;

	    texSolventParticles.channelDesc = cudaCreateChannelDesc<float2>();
	    texSolventParticles.filterMode = cudaFilterModePoint;
	    texSolventParticles.mipmapFilterMode = cudaFilterModePoint;
	    texSolventParticles.normalized = 0;
	    firsttime = false;
	}
	
	size_t textureoffset;
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texSolventParticles, solvent, &texSolventParticles.channelDesc,
				   sizeof(float) * 6 * npsolvent));

	const int ncells = L * L * L;
	
	assert(textureoffset == 0);
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texCellsStart, cellsstart, &texCellsStart.channelDesc, sizeof(int) * ncells));
	assert(textureoffset == 0);
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texCellsCount, cellscount, &texCellsCount.channelDesc, sizeof(int) * ncells));
	assert(textureoffset == 0);
    }
    
    __global__ void shift_send_particles(const Particle * const src, const int n, const int L, const int code, Particle * const dst)
    {
	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };
	
	if (gid < n)
	{
	    Particle p = src[gid];
	    
	    for(int c = 0; c < 3; ++c)
		p.x[c] -= d[c] * L;

	    dst[gid] = p;
	}
    }

    __device__ bool fsi_interaction(const int saru_tag,
				      const int dpid, const float3 xp, const float3 up, const int spid,
				      float& xforce, float& yforce, float& zforce)
    {
	xforce = yforce = zforce = 0;
	
	const int sentry = 3 * spid;
	
	const float2 stmp0 = tex1Dfetch(texSolventParticles, sentry);
	const float2 stmp1 = tex1Dfetch(texSolventParticles, sentry + 1);
	const float2 stmp2 = tex1Dfetch(texSolventParticles, sentry + 2);
	
	const float _xr = xp.x - stmp0.x;
	const float _yr = xp.y - stmp0.y;
	const float _zr = xp.z - stmp1.x;

	const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
	
	if (rij2 > 1)
	    return false;
	
	const float invrij = rsqrtf(rij2);
	
	const float rij = rij2 * invrij;
	const float wr = max((float)0, 1 - rij);
	
	const float xr = _xr * invrij;
	const float yr = _yr * invrij;
	const float zr = _zr * invrij;
	
	const float rdotv = 
	    xr * (up.x - stmp1.y) +
	    yr * (up.y - stmp2.x) +
	    zr * (up.z - stmp2.y);
	
	const float mysaru = saru(saru_tag, dpid, spid);
	const float myrandnr = 3.464101615f * mysaru - 1.732050807f;
	
	const float strength = (params.aij - params.gamma * wr * rdotv + params.sigmaf * myrandnr) * wr;
	
	xforce = strength * xr;
	yforce = strength * yr;
	zforce = strength * zr;

	return true;
    }

    __global__ void fsi_forces(const int saru_tag,
			       Acceleration * accsolvent, const int npsolvent,
			       const Particle * const particle, const int nparticles, Acceleration * accrbc, const int L)
    {
	const int dpid = threadIdx.x + blockDim.x * blockIdx.x;

	if (dpid >= nparticles)
	    return;

	const Particle p = particle[dpid];

	const float3 xp = make_float3(p.x[0], p.x[1], p.x[2]);
	const float3 up = make_float3(p.u[0], p.u[1], p.u[2]);
		
	int mycid[3];
	for(int c = 0; c < 3; ++c)
	    mycid[c] = (int)floor(p.x[c] + L/2);

	float fsum[3] = {0, 0, 0};
	
	for(int code = 0; code < 27; ++code)
	{
	    const int d[3] = {
		(code % 3) - 1,
		(code/3 % 3) - 1,
		(code/9 % 3) - 1
	    };
	    
	    int vcid[3];
	    for(int c = 0; c < 3; ++c)
		vcid[c] = mycid[c] + d[c];

	    bool validcid = true;
	    for(int c = 0; c < 3; ++c)
		validcid &= vcid[c] >= 0 && vcid[c] < L;

	    if (!validcid)
		continue;
	    
	    const int cid = vcid[0] + L * (vcid[1] + L * vcid[2]);
	    const int mystart = tex1Dfetch(texCellsStart, cid);
	    const int myend = mystart + tex1Dfetch(texCellsCount, cid);
	    
	    assert(mystart >= 0 && mystart <= myend);
	    assert(myend <= npsolvent);
	    
	    for(int s = mystart; s < myend; ++s)
	    {
		float f[3];
		const bool nonzero = fsi_interaction(saru_tag, dpid, xp, up, s, f[0], f[1], f[2]);

		if (nonzero)
		{
		    for(int c = 0; c < 3; ++c)
			fsum[c] += f[c];
		    
		    for(int c = 0; c < 3; ++c)
		    	atomicAdd(c + (float *)(accsolvent + s), -f[c]);
		}
	    }
	}
	
	for(int c = 0; c < 3; ++c)
	    accrbc[dpid].a[c] = fsum[c];
    }

    __global__ void merge_accelerations(const Acceleration * const src, const int n, Acceleration * const dst)
    {	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid < n)
	    for(int c = 0; c < 3; ++c)
		dst[gid].a[c] += src[gid].a[c];
    }
}

ComputeInteractionsRBC::ComputeInteractionsRBC(MPI_Comm _cartcomm, int L):  L(L), nvertices(CudaRBC::get_nvertices()), stream(0)
{
    assert(L % 2 == 0);
    assert(L >= 2);

    MPI_CHECK( MPI_Comm_dup(_cartcomm, &cartcomm));

    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
    MPI_CHECK( MPI_Comm_size(cartcomm, &nranks));

    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] + d[c];

	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i) );
    }

    KernelsRBC::ParamsFSI params = {aij, gammadpd, sigmaf};
    
    CUDA_CHECK(cudaMemcpyToSymbol(KernelsRBC::params, &params, sizeof(KernelsRBC::ParamsFSI)));
}

void ComputeInteractionsRBC::_compute_extents(const Particle * const rbcs, const int nrbcs)
{
    for(int i = 0; i < nrbcs; ++i)
	CudaRBC::extent_nohost(stream, (float *)(rbcs + nvertices * i), extents.devptr + i);
}

void ComputeInteractionsRBC::pack_and_post(const Particle * const rbcs, const int nrbcs)
{
    extents.resize(nrbcs);
 
    _compute_extents(rbcs, nrbcs);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    for(int i = 0; i < 26; ++i)
	haloreplica[i].clear();

    for(int i = 0; i < nrbcs; ++i)
    {
	const CudaRBC::Extent ext = extents.data[i];
	
	float pmin[3] = {ext.xmin, ext.ymin, ext.zmin};
	float pmax[3] = {ext.xmax, ext.ymax, ext.zmax};

	for(int code = 0; code < 26; ++code)
	{
	    int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };

	    bool interacting = true;
	    
	    for(int c = 0; c < 3; ++c)
	    {
		const float range_start = max((float)(d[c] * L - L/2 - 1), pmin[c]);
		const float range_end = min((float)(d[c] * L + L/2 + 1), pmax[c]);

		interacting &= range_end > range_start;
	    }

	    if (interacting)
		haloreplica[code].push_back(i);
	}
    }

    MPI_Request reqrecvcounts[26];
    for(int i = 0; i <26; ++i)
	MPI_CHECK(MPI_Irecv(recv_counts + i, 1, MPI_INTEGER, dstranks[i], recv_tags[i] + 2077, cartcomm, reqrecvcounts + i));

    MPI_Request reqsendcounts[26];
    for(int i = 0; i < 26; ++i)
    {
	send_counts[i] = haloreplica[i].size();
	MPI_CHECK(MPI_Isend(send_counts + i, 1, MPI_INTEGER, dstranks[i], i + 2077, cartcomm, reqsendcounts + i));
    }

    {
	MPI_Status statuses[26];
	MPI_CHECK(MPI_Waitall(26, reqrecvcounts, statuses));
	MPI_CHECK(MPI_Waitall(26, reqsendcounts, statuses));
    }

    for(int i = 0; i < 26; ++i)
	local[i].setup(send_counts[i] * nvertices);

    for(int i = 0; i < 26; ++i)
    {
	for(int j = 0; j < haloreplica[i].size(); ++j)
	    KernelsRBC::shift_send_particles<<< (nvertices + 127) / 128, 128, 0, stream>>>
		(rbcs + nvertices * haloreplica[i][j], nvertices, L, i, local[i].state.devptr + nvertices * j);
	 
	CUDA_CHECK(cudaPeekAtLastError());
    }
     
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for(int i = 0; i < 26; ++i)
	remote[i].setup(recv_counts[i] * nvertices);

    for(int i = 0; i < 26; ++i)
	if (recv_counts[i] > 0)
	{
	    MPI_Request request;
	    
	    MPI_CHECK(MPI_Irecv(remote[i].state.data, recv_counts[i] * nvertices, Particle::datatype(), dstranks[i],
				recv_tags[i] + 2011, cartcomm, &request));

	    reqrecvp.push_back(request);
	}

    for(int i = 0; i < 26; ++i)
	if (send_counts[i] > 0)
	{
	    MPI_Request request;

	    MPI_CHECK(MPI_Irecv(local[i].result.data, send_counts[i] * nvertices, Acceleration::datatype(), dstranks[i],
				recv_tags[i] + 2285, cartcomm, &request));

	    reqrecvacc.push_back(request);
	    
	    MPI_CHECK(MPI_Isend(local[i].state.data, send_counts[i] * nvertices, Particle::datatype(), dstranks[i],
				i + 2011, cartcomm, &request));

	    reqsendp.push_back(request);
	}
}

void ComputeInteractionsRBC::_internal_forces(const Particle * const rbcs, const int nrbcs, Acceleration * accrbc)
{
    for(int i = 0; i < nrbcs; ++i)
	CudaRBC::forces_nohost(stream, (float *)(rbcs + nvertices * i), (float *)(accrbc + nvertices * i));
}

void ComputeInteractionsRBC::evaluate(int& saru_tag,
				      const Particle * const solvent, const int nparticles, Acceleration * accsolvent,
				      const int * const cellsstart_solvent, const int * const cellscount_solvent,
				      const Particle * const rbcs, const int nrbcs, Acceleration * accrbc)
{	
    KernelsRBC::setup(solvent, nparticles, cellsstart_solvent, cellscount_solvent, L);

    pack_and_post(rbcs, nrbcs);

    if (nrbcs > 0 && nparticles > 0)
    {
	KernelsRBC::fsi_forces<<< (nrbcs * nvertices + 127) / 128, 128, 0, stream >>>
	    (saru_tag + myrank, accsolvent, nparticles, rbcs, nrbcs * nvertices, accrbc, L);
		
	_internal_forces(rbcs, nrbcs, accrbc);

	saru_tag += nranks;
    }
    
    _wait(reqrecvp);
    _wait(reqsendp);
    
    for(int i = 0; i < 26; ++i)
    {
	const int count = remote[i].state.size;

	if (count > 0)
	    KernelsRBC::fsi_forces<<< (count + 127) / 128, 128, 0, stream >>>
	    	(saru_tag + 26 * myrank + i, accsolvent, nparticles, remote[i].state.devptr, count, remote[i].result.devptr, L);
    }

    saru_tag += 26 * nranks;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    for(int i = 0; i < 26; ++i)
	if (recv_counts[i] > 0)
	{
	    MPI_Request request;
	    
	    MPI_CHECK(MPI_Isend(remote[i].result.data, recv_counts[i] * nvertices, Acceleration::datatype(), dstranks[i],
				i + 2285, cartcomm, &request));

	    reqsendacc.push_back(request);
	}

    _wait(reqrecvacc);
    _wait(reqsendacc);

    for(int i = 0; i < 26; ++i)
	for(int j = 0; j < haloreplica[i].size(); ++j)
	    KernelsRBC::merge_accelerations<<< (nvertices + 127) / 128, 128 >>>(local[i].result.devptr + nvertices * j, nvertices,
										accrbc + nvertices * haloreplica[i][j]);
}

ComputeInteractionsRBC::~ComputeInteractionsRBC()
{
    MPI_CHECK(MPI_Comm_free(&cartcomm));
}
