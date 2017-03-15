#include "stdio.h"

template <int k> struct Bspline {
  template <int i> static float eval(float x) {
    return (x - i) / (k - 1) * Bspline<k - 1>::template eval<i>(x) +
      (i + k - x) / (k - 1) * Bspline<k - 1>::template eval<i + 1>(x);
  }
  };

  template <> struct Bspline<1> {
    template <int i> static float eval(float x) {
      return (float)(i) <= x && x < (float)(i + 1);
    }
  };

float f0(float x) {
  Bspline<4> bsp;
  return bsp.eval<0>(x);
}

float f1(float x) {
  return \
    x <= 0 ? 0.0 :
    x <= 1 ? x*x*x/6 :
    x <= 2 ? (x*((12-3*x)*x-12)+4)/6 :
    x <= 3 ? (x*(x*(3*x-24)+60)-44)/6 :
    x <= 4 ? (x*((12-x)*x-48)+64)/6 :
    0.0;
}

int main() {
  int n = 100;
  float x, xlo = -1.0, xhi = 6.0, \
    L = xhi - xlo, dx = L/n;
  for (int i = 0; i < n; i++) {
    x = xlo + dx*i;
    printf("%g %g %g\n", x, f0(x), f1(x));    
  }
}
