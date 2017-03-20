#include "stdio.h"
#include <hd.h>

int main() {
  float a[3] = { 1,  2,  3};
  float b[3] = {10, 20, 30};
  float ans;
  fun(a, b, /**/ &ans);
  printf("ans: %g\n", ans);
}
