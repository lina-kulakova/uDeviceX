#include "hd.def.h"
#include "math.h"

#include "awall.up.h"

#define X 0
#define Y 1
#define Z 2
#define dt 0.1

namespace a {
  #define D Y
  #define rcx 0
  #define rcy 0
  #define rcz 0
  #define rcyl 2.0
  #include "awall.cpp"
}

namespace b {
  #define D Y
  #define rcx 0
  #define rcy 0
  #define rcz 4.0
  #define rcyl 2.0
  #include "awall.cpp"
}

__HD__ int bb(float *r0, float *v0, float *r1, float *v1) {
  int rcode;

  rcode = a::bb(r0, v0, r1, v1);
  if (rcode == BB_NORMAL) return rcode;

  rcode = b::bb(r0, v0, r1, v1);
  if (rcode == BB_NORMAL) return rcode;
  
  return rcode;
}

__HD__ bool inside(float *rg) {
  int rcode;

  rcode = a::inside(rg);
  if (rcode == true) return rcode;

  rcode = b::inside(rg);
  if (rcode == true) return rcode;

  return rcode;
}
