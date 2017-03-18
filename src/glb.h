namespace glb {
#define ndim 3
  extern __constant__  float r0[ndim];
  extern __constant__  float lg[ndim]; /* domain size (device) */
  extern               float LG[ndim]; /* domain size (host) */
#undef ndim
  void sim(); /* simulation-wide kernel globals */

  /* domain <-> subdomain coordinates transformation */
  void sub2dom(float*, float*);
  void dom2sub(float*, float*);
}
