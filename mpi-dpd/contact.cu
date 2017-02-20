#include <dpd-rng.h>

#include <vector>
#include <cstdio>
#include <mpi.h>
#include ".conf.h" /* configuration file (copy from .conf.test.h) */
#include "common.h"
#include "common-kernels.h"
#include "scan.h"
#include "contact.h"
#include "dpd-forces.h"
#include "last_bit_float.h"

#include "kernelscontact.decl.h"
#include "kernelscontact.impl.h"

ComputeContact::ComputeContact(MPI_Comm comm)
    : cellsstart(KernelsContact::NCELLS + 16),
      cellscount(KernelsContact::NCELLS + 16),
      compressed_cellscount(KernelsContact::NCELLS + 16) {
  int myrank;
  MPI_CHECK(MPI_Comm_rank(comm, &myrank));

  local_trunk = Logistic::KISS(7119 - myrank, 187 + myrank, 18278, 15674);

  CC(cudaPeekAtLastError());
}


void ComputeContact::build_cells(std::vector<ParticlesWrap> wsolutes,
                                 cudaStream_t stream) {
  this->nsolutes = wsolutes.size();

  int ntotal = 0;

  for (int i = 0; i < wsolutes.size(); ++i) ntotal += wsolutes[i].n;

  subindices.resize(ntotal);
  cellsentries.resize(ntotal);

  CC(cudaMemsetAsync(cellscount.D, 0, sizeof(int) * cellscount.S, stream));

  CC(cudaPeekAtLastError());

  int ctr = 0;
  for (int i = 0; i < wsolutes.size(); ++i) {
    ParticlesWrap it = wsolutes[i];

    if (it.n)
      subindex_local<true><<<(it.n + 127) / 128, 128, 0, stream>>>(
          it.n, (float2 *)it.p, cellscount.D, subindices.D + ctr);

    ctr += it.n;
  }

  compress_counts<<<(compressed_cellscount.S + 127) / 128, 128, 0, stream>>>(
      compressed_cellscount.S, (int4 *)cellscount.D,
      (uchar4 *)compressed_cellscount.D);

  scan(compressed_cellscount.D, compressed_cellscount.S, stream,
       (uint *)cellsstart.D);

  ctr = 0;
  for (int i = 0; i < wsolutes.size(); ++i) {
    ParticlesWrap it = wsolutes[i];

    if (it.n)
      KernelsContact::populate<<<(it.n + 127) / 128, 128, 0, stream>>>(
          subindices.D + ctr, cellsstart.D, it.n, i, ntotal,
          (KernelsContact::CellEntry *)cellsentries.D);

    ctr += it.n;
  }

  CC(cudaPeekAtLastError());

  KernelsContact::bind(cellsstart.D, cellsentries.D, ntotal, wsolutes, stream,
                       cellscount.D);
}

void ComputeContact::bulk(std::vector<ParticlesWrap> wsolutes,
                          cudaStream_t stream) {
  if (wsolutes.size() == 0) return;

  for (int i = 0; i < wsolutes.size(); ++i) {
    ParticlesWrap it = wsolutes[i];

    if (it.n)
      KernelsContact::bulk_3tpp<<<(3 * it.n + 127) / 128, 128, 0, stream>>>(
          (float2 *)it.p, it.n, cellsentries.S, wsolutes.size(), (float *)it.a,
          local_trunk.get_float(), i);

    CC(cudaPeekAtLastError());
  }
}

namespace KernelsContact {
__constant__ int packstarts_padded[27], packcount[26];
__constant__ Particle *packstates[26];
__constant__ Acceleration *packresults[26];

__global__ void halo(int nparticles_padded, int ncellentries,
                     int nsolutes, float seed) {
  int laneid = threadIdx.x & 0x1f;
  int warpid = threadIdx.x >> 5;
  int localbase = 32 * (warpid + 4 * blockIdx.x);
  int pid = localbase + laneid;

  if (localbase >= nparticles_padded) return;

  int nunpack;
  float2 dst0, dst1, dst2;
  float *dst = NULL;

  {
    uint key9 = 9 * (localbase >= packstarts_padded[9]) +
                      9 * (localbase >= packstarts_padded[18]);
    uint key3 = 3 * (localbase >= packstarts_padded[key9 + 3]) +
                      3 * (localbase >= packstarts_padded[key9 + 6]);
    uint key1 = (localbase >= packstarts_padded[key9 + key3 + 1]) +
                      (localbase >= packstarts_padded[key9 + key3 + 2]);
    int code = key9 + key3 + key1;
    int unpackbase = localbase - packstarts_padded[code];

    nunpack = min(32, packcount[code] - unpackbase);

    if (nunpack == 0) return;

    read_AOS6f((float2 *)(packstates[code] + unpackbase), nunpack, dst0, dst1,
               dst2);

    dst = (float *)(packresults[code] + unpackbase);
  }

  float xforce, yforce, zforce;
  read_AOS3f(dst, nunpack, xforce, yforce, zforce);

  int nzplanes = laneid < nunpack ? 3 : 0;

  for (int zplane = 0; zplane < nzplanes; ++zplane) {
    int scan1, scan2, ncandidates, spidbase;
    int deltaspid1, deltaspid2;

    {
      int xcenter = XOFFSET + (int)floorf(dst0.x);
      int xstart = max(0, xcenter - 1);
      int xcount = min(XCELLS, xcenter + 2) - xstart;

      if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0) continue;

      int ycenter = YOFFSET + (int)floorf(dst0.y);

      int zcenter = ZOFFSET + (int)floorf(dst1.x);
      int zmy = zcenter - 1 + zplane;
      bool zvalid = zmy >= 0 && zmy < ZCELLS;

      int count0 = 0, count1 = 0, count2 = 0;

      if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YCELLS) {
        int cid0 = xstart + XCELLS * (ycenter - 1 + YCELLS * zmy);
        spidbase = tex1Dfetch(texCellsStart, cid0);
        count0 = tex1Dfetch(texCellsStart, cid0 + xcount) - spidbase;
      }

      if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
        int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
        deltaspid1 = tex1Dfetch(texCellsStart, cid1);
        count1 = tex1Dfetch(texCellsStart, cid1 + xcount) - deltaspid1;
      }

      if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
        int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
        deltaspid2 = tex1Dfetch(texCellsStart, cid2);
        count2 = tex1Dfetch(texCellsStart, cid2 + xcount) - deltaspid2;
      }

      scan1 = count0;
      scan2 = count0 + count1;
      ncandidates = scan2 + count2;

      deltaspid1 -= scan1;
      deltaspid2 -= scan2;
    }

    for (int i = 0; i < ncandidates; ++i) {
      int m1 = (int)(i >= scan1);
      int m2 = (int)(i >= scan2);
      int slot = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

      CellEntry ce;
      ce.pid = tex1Dfetch(texCellEntries, slot);
      int soluteid = ce.code.w;
      ce.code.w = 0;

      int spid = ce.pid;

      int sentry = 3 * spid;
      float2 stmp0 = __ldg(csolutes[soluteid] + sentry);
      float2 stmp1 = __ldg(csolutes[soluteid] + sentry + 1);
      float2 stmp2 = __ldg(csolutes[soluteid] + sentry + 2);

      float myrandnr = Logistic::mean0var1(seed, pid, spid);

      // check for particle types and compute the DPD force
      float3 pos1 = make_float3(dst0.x, dst0.y, dst1.x),
             pos2 = make_float3(stmp0.x, stmp0.y, stmp1.x);
      float3 vel1 = make_float3(dst1.y, dst2.x, dst2.y),
             vel2 = make_float3(stmp1.y, stmp2.x, stmp2.y);
      int type1 = MEMB_TYPE;
      int type2 = MEMB_TYPE;
      float3 strength = compute_dpd_force_traced(type1, type2, pos1, pos2,
                                                       vel1, vel2, myrandnr);

      float xinteraction = strength.x;
      float yinteraction = strength.y;
      float zinteraction = strength.z;

      xforce += xinteraction;
      yforce += yinteraction;
      zforce += zinteraction;

      atomicAdd(csolutesacc[soluteid] + sentry, -xinteraction);
      atomicAdd(csolutesacc[soluteid] + sentry + 1, -yinteraction);
      atomicAdd(csolutesacc[soluteid] + sentry + 2, -zinteraction);
    }
  }

  write_AOS3f(dst, nunpack, xforce, yforce, zforce);
}
}

void ComputeContact::halo(ParticlesWrap halos[26], cudaStream_t stream) {
  int nremote_padded = 0;

  {
    int recvpackcount[26], recvpackstarts_padded[27];

    for (int i = 0; i < 26; ++i) recvpackcount[i] = halos[i].n;

    CC(cudaMemcpyToSymbolAsync(KernelsContact::packcount, recvpackcount,
                               sizeof(recvpackcount), 0, cudaMemcpyHostToDevice,
                               stream));

    recvpackstarts_padded[0] = 0;
    for (int i = 0, s = 0; i < 26; ++i)
      recvpackstarts_padded[i + 1] = (s += 32 * ((halos[i].n + 31) / 32));

    nremote_padded = recvpackstarts_padded[26];

    CC(cudaMemcpyToSymbolAsync(
        KernelsContact::packstarts_padded, recvpackstarts_padded,
        sizeof(recvpackstarts_padded), 0, cudaMemcpyHostToDevice, stream));

    const Particle *recvpackstates[26];

    for (int i = 0; i < 26; ++i) recvpackstates[i] = halos[i].p;

    CC(cudaMemcpyToSymbolAsync(KernelsContact::packstates, recvpackstates,
                               sizeof(recvpackstates), 0,
                               cudaMemcpyHostToDevice, stream));
    Acceleration *packresults[26];
    for (int i = 0; i < 26; ++i) packresults[i] = halos[i].a;
    CC(cudaMemcpyToSymbolAsync(KernelsContact::packresults, packresults,
                               sizeof(packresults), 0, cudaMemcpyHostToDevice,
                               stream));
  }

  if (nremote_padded)
    KernelsContact::halo<<<(nremote_padded + 127) / 128, 128, 0, stream>>>(
        nremote_padded, cellsentries.S, nsolutes, local_trunk.get_float());
  CC(cudaPeekAtLastError());
}
