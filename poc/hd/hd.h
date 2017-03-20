#ifdef __CUDACC__
  #define __HD__ __host__ __device__
#else
  #define __HD__
#endif

__HD__ void fun(float*, float*, /**/ float*);
