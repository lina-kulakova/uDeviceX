/* type of bounce back outcome */
enum {BB_NO,
      BB_RO_INSIDE, /* r0 was already inside */
      BB_NORMAL,
      BB_RESCUE,
      BB_FAIL};

extern float Rw[3];

/* solve a*x^2 + 2*k*x + c, returns the number of roots */
__HD__ int  solve_half_quadratic(float a, float k, float c, /**/ float *x0, float *x1);

__HD__ int bb(float *R0_, float *V0_,
	      /*inout*/ float *R1_, float *V1_);
