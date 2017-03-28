#include "hd.def.h"
#include ".conf.h"
#include "awall.up.h"
#include "math.h"

#ifdef cy0_D
namespace cy0 {
  #define    D cy0_D
  #define  rcx cy0_rcx
  #define  rcy cy0_rcy
  #define  rcz cy0_rcz
  #define rcyl cy0_rcyl

  #include "cy.h"

  #undef D
  #undef rcx
  #undef rcy
  #undef rcz
  #undef rcyl
}
#endif

#ifdef cy1_D
namespace cy1 {
  #define    D cy1_D
  #define  rcx cy1_rcx
  #define  rcy cy1_rcy
  #define  rcz cy1_rcz
  #define rcyl cy1_rcyl

  #include "cy.h"

  #undef D
  #undef rcx
  #undef rcy
  #undef rcz
  #undef rcyl
}
#endif

#ifdef pl0_rcx
namespace pl0 {
  #define rcx pl0_rcx
  #define rcy pl0_rcy
  #define rcz pl0_rcz
  #define  nx pl0_nx
  #define  ny pl0_ny
  #define  nz pl0_nz

  #include "pl.h"

  #undef rcx
  #undef rcy
  #undef rcz
  #undef nx
  #undef ny
  #undef nz
}
#endif

#ifdef pl1_rcx
namespace pl1 {
  #define rcx pl1_rcx
  #define rcy pl1_rcy
  #define rcz pl1_rcz
  #define  nx pl1_nx
  #define  ny pl1_ny
  #define  nz pl1_nz

  #include "pl.h"

  #undef rcx
  #undef rcy
  #undef rcz
  #undef nx
  #undef ny
  #undef nz
}
#endif

__HD__ int bb(float *r0, float *v0, float *r1, float *v1) {
  int rcode = BB_NO;

#ifdef cy0_D
  rcode = cy0::bb(r0, v0, r1, v1);
  if (rcode == BB_NORMAL) return rcode;
#endif

#ifdef cy1_D
  rcode = cy1::bb(r0, v0, r1, v1);
  if (rcode == BB_NORMAL) return rcode;
#endif

#ifdef pl0_rcx
  rcode = pl0::bb(r0, v0, r1, v1);
  if (rcode == BB_NORMAL) return rcode;
#endif

#ifdef pl1_rcx
  rcode = pl1::bb(r0, v0, r1, v1);
  if (rcode == BB_NORMAL) return rcode;
#endif

  return rcode;
}

__HD__ bool inside(float *rg) {
  int rcode = false;

#ifdef cy0_D
  rcode = cy0::inside(rg);
  if (rcode) return rcode;
#endif

#ifdef cy1_D
  rcode = cy1::inside(rg);
  if (rcode) return rcode;
#endif

#ifdef pl0_rcx
  rcode = pl0::inside(rg);
  if (rcode) return rcode;
#endif

#ifdef pl1_rcx
  rcode = pl1::inside(rg);
  if (rcode) return rcode;
#endif

  return rcode;
}
