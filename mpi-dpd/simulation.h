/*
 *  simulation.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-03-24.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */



class Simulation
{
  ParticleArray particles_pingpong[2];
  ParticleArray * particles, * newparticles;
  SimpleDeviceBuffer<float4> xyzouvwo;
  SimpleDeviceBuffer<ushort4> xyzo_half;

  CellLists* cells;
  CollectionRBC * rbcscoll;

  RedistributeParticles redistribute;
  RedistributeRBCs redistribute_rbcs;

  ComputeDPD dpd;
  SoluteExchange solutex;
  ComputeFSI fsi;
  ComputeContact* contact;

  ComputeWall * wall;
  bool simulation_is_done;

  MPI_Comm activecomm, cartcomm;
  cudaStream_t mainstream, uploadstream, downloadstream;

  size_t nsteps;
  float driving_acceleration;
  int nranks, rank;

  void _update_helper_arrays();

  void _redistribute();
  void _create_walls();
  void _remove_bodies_from_wall(CollectionRBC * coll);
  void _forces();
  void _datadump(const int idtimestep);
  void _update_and_bounce();
  void _lockstep();

  pthread_t thread_datadump;
  pthread_mutex_t mutex_datadump;
  pthread_cond_t request_datadump, done_datadump;
  bool datadump_pending;
  int datadump_idtimestep, datadump_nsolvent, datadump_nrbcs;
  bool async_thread_initialized;

  PinnedHostBuffer<Particle> particles_datadump;
  PinnedHostBuffer<Acceleration> accelerations_datadump;

  cudaEvent_t evdownloaded;

  void  _datadump_async();

 public:

  Simulation(MPI_Comm cartcomm, MPI_Comm activecomm) ;

  void run();

  ~Simulation();

  static void * datadump_trampoline(void * x) { ((Simulation *)x)->_datadump_async(); return NULL; }
};
