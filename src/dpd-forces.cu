#include "dpd-forces.h"
#include <cstdio>
#include <mpi.h>
#include <utility>
#include <cell-lists.h>
#include <cuda-dpd.h>
#include ".conf.h" /* configuration file (copy from .conf.test.h) */
#include "common.h"
#include "common.tmp.h"

__device__ float3 compute_dpd_force_traced(int type1, int type2,
        float3 pos1, float3 pos2, float3 vel1, float3 vel2, float myrandnr) {
  /* return the DPD interaction force based on particle types */

    float gammadpd[] = {_gammadpd_out, _gammadpd_in, _gammadpd_wall};
    float aij[] = {_aij_out / rc, _aij_in / rc, _aij_wall / rc};

    float _xr = pos1.x - pos2.x;
    float _yr = pos1.y - pos2.y;
    float _zr = pos1.z - pos2.z;

    float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
    float invrij = rsqrtf(rij2);
    float rij = rij2 * invrij;
    if (rij2 >= 1)
        return make_float3(0, 0, 0);

    float argwr = 1.f - rij;
    float wr = viscosity_function<-VISCOSITY_S_LEVEL>(argwr);

    float xr = _xr * invrij;
    float yr = _yr * invrij;
    float zr = _zr * invrij;

    float rdotv =
        xr * (vel1.x - vel2.x) +
        yr * (vel1.y - vel2.y) +
        zr * (vel1.z - vel2.z);

    float gammadpd_pair = 0.5 * (gammadpd[type1] + gammadpd[type2]);
    float sigmaf_pair = sqrt(2*gammadpd_pair*kBT / dt);
    float strength = (-gammadpd_pair * wr * rdotv + sigmaf_pair * myrandnr) * wr;
    float aij_pair = 0.5 * (aij[type1] + aij[type2]);
    
    strength += aij_pair * argwr;

    return make_float3(strength*xr, strength*yr, strength*zr);
}
