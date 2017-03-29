#include <cstdio>
#include <mpi.h>
#include <map>
#include ".conf.h" /* configuration file (copy from .conf.test.h) */
#include "m.h"     /* MPI */
#include "common.h"
#include "bund.h"
#include "glb.h"

int main(int argc, char **argv) {
  m::dims[0] = m::dims[1] = m::dims[2] = 1;
  for (int iarg = 1; iarg < argc && iarg <= 3; iarg++)
    m::dims[iarg - 1] = atoi(argv[iarg]);

  int device = 0;
  CC(cudaSetDevice(device));

  m::init(argc, argv); /* MPI init */
  glb::sim(); /* simulation level globals */

  sim::init();
  sim::run();
  sim::close();
  
  MPI_Finalize();
}
