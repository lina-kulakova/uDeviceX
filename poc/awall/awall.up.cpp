#include "hd.def.h"
#include "math.h"

#include "awall.up.h"

#define X 0
#define Y 1
#define Z 2
#define dt 0.1

namespace c {
  #define D Y
  #define rcx 0
  #define rcy 5.0
  #define rcz 0
  #define rcyl 8.0
  #include "awall.cpp"
}

__HD__ int bb(float *r0, float *v0, float *r1, float *v1) {
  int rcode;

  rcode = c::bb(r0, v0, r1, v1);
  if (rcode == BB_NORMAL) return rcode;
  
  return rcode;
}

__HD__ bool inside(float *rg) {
  int rcode;

  rcode = c::inside(rg);
  if (rcode == true) return rcode;

  return rcode;
}
