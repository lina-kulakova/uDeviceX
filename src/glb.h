namespace glb {
  extern __constant__  float r0[3];
  extern __constant__  float lg[3]; /* domain size (device) */
  extern               float LG[3]; /* domain size (host) */
  void sim(); /* simulation-wide kernel globals */

  /* domain <-> subdomain coordinates transformation */
  void sub2dom(float*, float*);
  void dom2sub(float*, float*);
}
