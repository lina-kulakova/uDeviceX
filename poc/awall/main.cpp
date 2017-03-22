#include "hd.def.h"
#include "awall.h"
#include <stdio.h>

/*
 in   :  R0, V0, vwall(R)
 inout:  R1, V1
 out  :  dP, dL
*/

#define dt 0.1

int main() {
  float a = 1e-12, k = 2, c = 1e-6 + 1;
  float x0, x1;
  int n = solve_half_quadratic(a, k, c, &x0, &x1);

  printf("n = %d\n", n);
  printf("roots: %.4g %.4g\n", x0, x1);
}
