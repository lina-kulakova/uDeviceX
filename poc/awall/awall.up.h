/* type of bounce back outcome */
enum {BB_NO,
      BB_R0_INSIDE, /* r0 was already inside */
      BB_NORMAL,
      BB_RESCUE,
      BB_FAIL};

__HD__ int  bb(float *r0, float *v0, float *r1, float *v1);
__HD__ bool inside(float *rg);
