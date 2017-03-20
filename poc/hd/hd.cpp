#include "hd.h"

__HD__ void fun(float *a, float *b, /**/ float *ans) {
  enum {X, Y, Z};
  *ans = a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z];
}
