#include <stdio.h>

void f(float *a) {
  printf("a: %g %g %g\n", a[0], a[1], a[10]);
}

int main() {
  float a[3] = {0, 10, 20};
  f(a);
}
