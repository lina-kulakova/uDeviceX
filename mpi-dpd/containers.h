/*
 *  containers.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-05.
 *  Copyright 2015. All rights reserved. */

struct ParticleArray {
  int S; /* size */
  SimpleDeviceBuffer<Particle>     pp; /* xyzuvw */
  SimpleDeviceBuffer<Acceleration> aa; /* axayaz */

  void pa_resize(int n);
  void pa_preserve_resize(int n);
  void upd_stg1      (bool rbcflag, float driving_acceleration, cudaStream_t stream);
  void upd_stg2_and_1(bool rbcflag, float driving_acceleration, cudaStream_t stream);
  void clear_velocity();

  void clear_acc(cudaStream_t stream) {
    CC(cudaMemsetAsync(aa.D, 0, sizeof(Acceleration) * aa.S, stream));
  }
};

void rbc_dump(MPI_Comm comm,
	      Particle* p, Acceleration* a, int n, int iddatadump);

/*** put in simulation.cu ****/
extern int nvertices;
extern MPI_Comm cartcomm;
extern int rank;
extern int ncells;
/***                      ***/

class CollectionRBC : public ParticleArray {
 protected:
  void _initialize(float *device_pp, float (*transform)[4]);
 public:
  CollectionRBC();
  void setup(const char *path2ic);
  void remove(int*  entries, int nentries);
  void rbc_resize(int rbcs_count);
  void rbc_preserve_resize(int n);
  int  pcount() {return ncells * nvertices;}
};
