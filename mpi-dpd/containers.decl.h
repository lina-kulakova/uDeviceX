namespace Cont {
  int (*indices)[3] = NULL;
  int nt = -1;
  int nv = -1;
  float3 origin, globalextent;
  int    coords[3];
  int nranks, rank;
  int nc = 0;
  struct TransformedExtent {
    float com[3];
    float transform[4][4];
  };
}
