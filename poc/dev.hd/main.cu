#include <stdio.h>

#include <hd.def.h>
#include <hd.h>

__global__ void
main_dev(float *a, float *b, float *ans) {
  fun(a, b, ans);
}

int main() {
  float a[] = { 1,  2,  3};
  float b[] = {10, 20, 30};
  float ans;

  float *a_dev, *b_dev, *ans_dev;
  int sz1 = sizeof(ans), sz3 = sizeof(a);

  cudaMalloc(&a_dev, sz3); cudaMalloc(&b_dev, sz3);
  cudaMalloc(&ans_dev, sz1);

  cudaMemcpy(a_dev, a, sz3, cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b, sz3, cudaMemcpyHostToDevice);
  main_dev<<<1,1>>>(a_dev, b_dev, ans_dev);
  cudaMemcpy(&ans, ans_dev, sz1, cudaMemcpyDeviceToHost);

  printf("ans: %g\n", ans);
}
