#include <mpi.h>
#include "m.h"

namespace m { /* MPI (man MPI_Cart_get) */
  const int d = 3;
  int periods[d] = {true, true, true};
  /* set in main */
  int  rank, coords[d], dims[d];
  const bool reorder = false;
  MPI_Comm cart;


  void init(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,   &rank);
    MPI_Cart_create(MPI_COMM_WORLD,
		       d, dims, periods, reorder,   &cart);
    MPI_Cart_coords(cart, rank, d,   coords);
}

}
