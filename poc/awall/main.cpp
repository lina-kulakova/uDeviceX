#include <stdio.h>
#include <stdlib.h>

#include "hd.def.h"
#include "awall.h"
#define dt 0.1

/*
 in   :  R0, V0, vwall(R)
 inout:  R1, V1
 out  :  dP, dL
*/


int main(int argc, char* argv[]) {
  enum {X, Y, Z};
  int c;

  float R0[3], V0[3];
  for (c = 0; c < 3; c++) R0[c] = atof(*(++argv));
  for (c = 0; c < 3; c++) V0[c] = atof(*(++argv));  

  float R1[3], V1[3];
  for (c = 0; c < 3; c++) R1[c] = R0[c] + V0[c]*dt;
  for (c = 0; c < 3; c++) V1[c] = V0[c];

  printf( "%g %g %g ", R0[X], R0[Y], R0[Z]);
  printf( "%g %g %g ", R1[X], R1[Y], R1[Z]);
  int code = bb(R0, V0, /**/ R1, V1);
  printf("%g %g %g ", R1[X], R1[Y], R1[Z]);
  printf("%d\n"     , code);
}
