enum {X, Y, Z};

enum {BB_NO, /* type of bounce back */
      BB_NORMAL,
      BB_RESCUE,
      BB_FAIL};

/* solve a*x^2 + 2*k*x + c, returns the number of roots */
__HD__ int  solve_half_quadratic(float a, float k, float c, /**/ float *x0, float * x1);
