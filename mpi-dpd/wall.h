/*
 *  wall.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-19.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */



namespace SolidWallsKernel
{
__global__ void fill_keys(const Particle * const particles, const int n, int * const key);
}

class ComputeWall
{
  MPI_Comm cartcomm;
  int myrank, dims[3], periods[3], coords[3];

  Logistic::KISS trunk;

  int solid_size;
  float4 * solid4;

  cudaArray * arrSDF;

  CellLists* wall_cells;

  SimpleDeviceBuffer<float3> frcs;
  int samples;

 public:

  ComputeWall(MPI_Comm cartcomm, Particle* const p, const int n, int& nsurvived, ExpectedMessageSizes& new_sizes);

  ~ComputeWall();

  void wall_bounce(Particle * const p, const int n, cudaStream_t stream);

  void wall_interactions(const Particle * const p, const int n, Acceleration * const acc,
                    cudaStream_t stream);
};
