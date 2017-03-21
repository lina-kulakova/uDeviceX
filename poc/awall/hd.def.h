#ifdef __CUDACC__
  #define __H__  __host__
  #define __D__  __device__
  #define __HD__ __H__ __D__
#else
  #define __H__
  #define __D__
  #define __HD__
#endif
