/* type of bounce back outcome */
enum {BB_NO,
      BB_R0_INSIDE, /* r0 was already inside */
      BB_NORMAL,
      BB_RESCUE,
      BB_FAIL};

extern float Rw[3];

__HD__ int  bb(float *R0_, float *V0_, /*inout*/ float *R1_, float *V1_);
__HD__ bool inside(float *R);
