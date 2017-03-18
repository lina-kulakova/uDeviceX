#include <mpi.h>
#include ".conf.h"
#include "common.h"
#include "m.h"
#include "glb.h"

#define ndim 3

/* global variables visible for every kernel */
namespace glb {
  enum {X, Y, Z};
  __constant__ float r0[ndim];
  __constant__ float lg[ndim];
  float              LG[ndim];

  /* subdomain to domain coordinates */
  void sub2dom(float *r, /**/ float *q) {
    q[X] = r[X] + XS*(m::coords[X] + 0.5);
    q[Y] = r[Y] + XS*(m::coords[Y] + 0.5);
    q[Z] = r[Z] + XS*(m::coords[Z] + 0.5);
  }

  /* domain to subdomain coordinates */
  void dom2sub(float *r, /**/ float *q) {
    q[X] = r[X] - XS*(m::coords[X] + 0.5);
    q[Y] = r[Y] - XS*(m::coords[Y] + 0.5);
    q[Z] = r[Z] - XS*(m::coords[Z] + 0.5);
  }

  void sim() {
    /* all coordinates are relative to the center of the sub-domain;
       Example: (dims[X] = 3, `XS' is sub-domain size):
	 |            |             |             |
      -XS/2          XS/2        3XS/2         5XS/2
	  coords[X]=0   coords[X]=1   coords[X]=2
     */

    float r0_h[ndim]; /* the center of the domain in sub-domain
		     coordinates; to go to domain coordinates (`rg')
		     from sub-domain coordinates (`r'): rg = r - r0
		   */
    r0_h[X] = XS*(m::dims[X]-2*m::coords[X]-1)*0.5;
    r0_h[Y] = YS*(m::dims[Y]-2*m::coords[Y]-1)*0.5;
    r0_h[Z] = ZS*(m::dims[Z]-2*m::coords[Z]-1)*0.5;
    cudaMemcpyToSymbol(r0, r0_h, ndim*sizeof(float));

    LG[X] = XS*m::dims[X];
    LG[Y] = YS*m::dims[Y];
    LG[Z] = ZS*m::dims[Z];
    cudaMemcpyToSymbol(lg, LG, ndim*sizeof(float));
  }
}
#undef ndim
