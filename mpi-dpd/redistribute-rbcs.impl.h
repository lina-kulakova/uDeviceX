namespace RedistRBC {
void _post_recvcount() {
  recv_counts[0] = 0;
  for (int i = 1; i < 27; ++i) {
    MPI_Request req;
    MC(MPI_Irecv(recv_counts + i, 1, MPI_INTEGER, anti_rankneighbors[i],
			i + 1024, cartcomm, &req));
    recvcountreq.push_back(req);
  }
}

void redistribute_rbcs_init(MPI_Comm _cartcomm) {
  bulk = new DeviceBuffer<Particle>;
  for (int i = 0; i < HALO_BUF_SIZE; i++) halo_recvbufs[i] = new PinnedHostBuffer<Particle>;
  for (int i = 0; i < HALO_BUF_SIZE; i++) halo_sendbufs[i] = new PinnedHostBuffer<Particle>;
  minextents = new PinnedHostBuffer<float3>;
  maxextents = new PinnedHostBuffer<float3>;
    
  nvertices = CudaRBC::get_nvertices();
  CudaRBC::Extent host_extent;
  CudaRBC::setup(nvertices, host_extent);
  /* TODO: move it to a better place; [xyz]lo, [xyz]hi pbc[xyz] (9
     arguments for iotags_domain, pbc: 1: for periods boundary
     conditions) */
  iotags_init_file("../cuda-rbc/rbc.dat");
  iotags_domain(0, 0, 0,
		XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN,
		1, 1, 1);
  MC(MPI_Comm_dup(_cartcomm, &cartcomm));
  MC(MPI_Comm_rank(cartcomm, &myrank));
  int dims[3];
  MC(MPI_Cart_get(cartcomm, 3, dims, periods, coords));

  rankneighbors[0] = myrank;
  for (int i = 1; i < 27; ++i) {
    int d[3] = {(i + 1) % 3 - 1, (i / 3 + 1) % 3 - 1, (i / 9 + 1) % 3 - 1};
    int coordsneighbor[3];
    for (int c = 0; c < 3; ++c) coordsneighbor[c] = coords[c] + d[c];
    MC(MPI_Cart_rank(cartcomm, coordsneighbor, rankneighbors + i));
    for (int c = 0; c < 3; ++c) coordsneighbor[c] = coords[c] - d[c];
    MC(MPI_Cart_rank(cartcomm, coordsneighbor, anti_rankneighbors + i));
  }

  CC(cudaEventCreate(&evextents, cudaEventDisableTiming));
  _post_recvcount();
}

void _compute_extents(Particle *xyzuvw,
					int nrbcs, cudaStream_t stream) {
  if (nrbcs)
    minmax(xyzuvw, nvertices, nrbcs, minextents->devptr, maxextents->devptr,
	   stream);
}

namespace ReorderingRBC {
static const int cmaxnrbcs = 64 * 4;
__constant__ float *csources[cmaxnrbcs], *cdestinations[cmaxnrbcs];

template <bool from_cmem>
__global__ void pack_all_kernel(const int nrbcs, const int nvertices,
				const float **const dsources,
				float **const ddestinations) {
  if (nrbcs == 0) return;
  const int nfloats_per_rbc = 6 * nvertices;
  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid >= nfloats_per_rbc * nrbcs) return;
  const int idrbc = gid / nfloats_per_rbc;
  const int offset = gid % nfloats_per_rbc;

  float val;
  if (from_cmem) val = csources[idrbc][offset];
  else           val = dsources[idrbc][offset];

  if (from_cmem) cdestinations[idrbc][offset] = val;
  else           ddestinations[idrbc][offset] = val;

}

void pack_all(cudaStream_t stream, const int nrbcs, const int nvertices,
	      const float **const sources, float **const destinations) {
  if (nrbcs == 0) return;
  int nthreads = nrbcs * nvertices * 6;

  if (nrbcs < cmaxnrbcs) {
    CC(cudaMemcpyToSymbolAsync(cdestinations, destinations,
			       sizeof(float *) * nrbcs, 0,
			       cudaMemcpyHostToDevice, stream));
    CC(cudaMemcpyToSymbolAsync(csources, sources, sizeof(float *) * nrbcs, 0,
			       cudaMemcpyHostToDevice, stream));
    pack_all_kernel<true><<<(nthreads + 127) / 128, 128, 0, stream>>>(
	nrbcs, nvertices, NULL, NULL);
  } else {
    _ddestinations.resize(nrbcs);
    _dsources.resize(nrbcs);
    CC(cudaMemcpyAsync(_ddestinations.D, destinations, sizeof(float *) * nrbcs,
		       cudaMemcpyHostToDevice, stream));
    CC(cudaMemcpyAsync(_dsources.D, sources, sizeof(float *) * nrbcs,
		       cudaMemcpyHostToDevice, stream));
    pack_all_kernel<false><<<(nthreads + 127) / 128, 128, 0, stream>>>(
	nrbcs, nvertices, _dsources.D, _ddestinations.D);
  }
  CC(cudaPeekAtLastError());
}
}

void extent(Particle *xyzuvw, int nrbcs,
			      cudaStream_t stream) {
  minextents->resize(nrbcs);
  maxextents->resize(nrbcs);
  CC(cudaPeekAtLastError());
  _compute_extents(xyzuvw, nrbcs, stream);
  CC(cudaPeekAtLastError());
  CC(cudaEventRecord(evextents, stream));
}

void pack_sendcount(Particle *xyzuvw,
				      int nrbcs, cudaStream_t stream) {
  CC(cudaEventSynchronize(evextents));
  std::vector<int> reordering_indices[27];

  for (int i = 0; i < nrbcs; ++i) {
    float3 minext = minextents->data[i];
    float3 maxext = maxextents->data[i];
    float p[3] = {0.5 * (minext.x + maxext.x), 0.5 * (minext.y + maxext.y),
		  0.5 * (minext.z + maxext.z)};
    int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
    int vcode[3];
    for (int c = 0; c < 3; ++c)
      vcode[c] = (2 + (p[c] >= -L[c] / 2) + (p[c] >= L[c] / 2)) % 3;
    int code = vcode[0] + 3 * (vcode[1] + 3 * vcode[2]);
    reordering_indices[code].push_back(i);
  }

  bulk->resize(reordering_indices[0].size() * nvertices);
  for (int i = 1; i < 27; ++i)
    halo_sendbufs[i]->resize(reordering_indices[i].size() * nvertices);
  {
    static std::vector<const float *> src;
    static std::vector<float *> dst;
    src.clear();
    dst.clear();
    for (int i = 0; i < 27; ++i)
      for (int j = 0; j < reordering_indices[i].size(); ++j) {
	src.push_back((float *)(xyzuvw + nvertices * reordering_indices[i][j]));

	if (i)
	  dst.push_back((float *)(halo_sendbufs[i]->devptr + nvertices * j));
	else
	  dst.push_back((float *)(bulk->D + nvertices * j));
      }
    ReorderingRBC::pack_all(stream, src.size(), nvertices, &src.front(),
			    &dst.front());
    CC(cudaPeekAtLastError());
  }
  CC(cudaStreamSynchronize(stream));
  for (int i = 1; i < 27; ++i)
    MC(MPI_Isend(&halo_sendbufs[i]->size, 1, MPI_INTEGER,
			rankneighbors[i], i + 1024, cartcomm,
			&sendcountreq[i - 1]));
}

int post() {
  {
    MPI_Status statuses[recvcountreq.size()];
    MC(MPI_Waitall(recvcountreq.size(), &recvcountreq.front(), statuses));
    recvcountreq.clear();
  }

  arriving = 0;
  for (int i = 1; i < 27; ++i) {
    int count = recv_counts[i];
    arriving += count;
    halo_recvbufs[i]->resize(count);
  }

  arriving /= nvertices;
  notleaving = bulk->S / nvertices;

  MPI_Status statuses[26];
  MC(MPI_Waitall(26, sendcountreq, statuses));

  for (int i = 1; i < 27; ++i)
    if (halo_recvbufs[i]->size > 0) {
      MPI_Request request;
      MC(MPI_Irecv(halo_recvbufs[i]->data, halo_recvbufs[i]->size,
			  Particle::datatype(), anti_rankneighbors[i], i + 1155,
			  cartcomm, &request));
      recvreq.push_back(request);
    }

  for (int i = 1; i < 27; ++i)
    if (halo_sendbufs[i]->size > 0) {
      MPI_Request request;
      MC(MPI_Isend(halo_sendbufs[i]->data, halo_sendbufs[i]->size,
			  Particle::datatype(), rankneighbors[i], i + 1155,
			  cartcomm, &request));

      sendreq.push_back(request);
    }
  return notleaving + arriving;
}

namespace ParticleReorderingRBC {
__global__ void shift(const Particle *const psrc, const int np, const int code,
		      const int rank, const bool check, Particle *const pdst) {
  int pid = threadIdx.x + blockDim.x * blockIdx.x;
  int d[3] = {(code + 1) % 3 - 1, (code / 3 + 1) % 3 - 1,
	      (code / 9 + 1) % 3 - 1};
  if (pid >= np) return;
  Particle pnew = psrc[pid];
  int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
  for (int c = 0; c < 3; ++c) pnew.x[c] -= d[c] * L[c];
  pdst[pid] = pnew;
}
}

void unpack(Particle *xyzuvw, int nrbcs,
			      cudaStream_t stream) {
  MPI_Status statuses[26];
  MC(MPI_Waitall(recvreq.size(), &recvreq.front(), statuses));
  MC(MPI_Waitall(sendreq.size(), &sendreq.front(), statuses));
  recvreq.clear();
  sendreq.clear();
  CC(cudaMemcpyAsync(xyzuvw, bulk->D, notleaving * nvertices * sizeof(Particle),
		     cudaMemcpyDeviceToDevice, stream));

  for (int i = 1, s = notleaving * nvertices; i < 27; ++i) {
    int count = halo_recvbufs[i]->size;
    if (count > 0)
      ParticleReorderingRBC::shift<<<(count + 127) / 128, 128, 0, stream>>>(
	  halo_recvbufs[i]->devptr, count, i, myrank, false, xyzuvw + s);
    s += halo_recvbufs[i]->size;
  }

  CC(cudaPeekAtLastError());

  _post_recvcount();
}

void redistribute_rbcs_close() {
  MC(MPI_Comm_free(&cartcomm));
  delete bulk;
  for (int i = 0; i < HALO_BUF_SIZE; i++) delete halo_recvbufs[i];
  for (int i = 0; i < HALO_BUF_SIZE; i++) delete halo_sendbufs[i];
  delete minextents;
  delete maxextents;
}

}
