#include <mpi.h>
#include <sys/resource.h>
#include <utility>
#include <cell-lists.h>
#include <cuda-dpd.h>
#include <cstdio>
#include ".conf.h" /* configuration file (copy from .conf.test.h) */
#include "common.h"

bool Particle::initialized = false;

MPI_Datatype Particle::mytype;

bool Acceleration::initialized = false;

MPI_Datatype Acceleration::mytype;

void CellLists::build(Particle * const p, const int n, cudaStream_t stream, int * const order, const Particle * const src)
{
    if (n > 0)
      build_clists_vanilla((float * )p, n, 1, LX, LY, LZ, -LX/2, -LY/2, -LZ/2, order, start, count,  NULL, stream, (float *)src);
    else
    {
        CC(cudaMemsetAsync(start, 0, sizeof(int) * ncells, stream));
        CC(cudaMemsetAsync(count, 0, sizeof(int) * ncells, stream));
    }
}


void diagnostics(MPI_Comm comm, MPI_Comm cartcomm, Particle * particles, int n, float dt, int idstep, Acceleration * acc)
{
    double p[] = {0, 0, 0};
    for(int i = 0; i < n; ++i)
        for(int c = 0; c < 3; ++c)
            p[c] += particles[i].u[c];

    int rank;
    MC( MPI_Comm_rank(comm, &rank) );

    int dims[3], periods[3], coords[3];
    MC( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    MC( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &p, rank == 0 ? &p : NULL, 3, MPI_DOUBLE, MPI_SUM, 0, comm) );

    double ke = 0;
    for(int i = 0; i < n; ++i)
        ke += pow(particles[i].u[0], 2) + pow(particles[i].u[1], 2) + pow(particles[i].u[2], 2);

    MC( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &ke, &ke, 1, MPI_DOUBLE, MPI_SUM, 0, comm) );
    MC( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &n, &n, 1, MPI_INT, MPI_SUM, 0, comm) );

    double kbt = 0.5 * ke / (n * 3. / 2);

    if (rank == 0)
    {
        static bool firsttime = true;
        FILE * f = fopen("diag.txt", firsttime ? "w" : "a");
        firsttime = false;

        if (idstep == 0)
            fprintf(f, "# TSTEP\tKBT\tPX\tPY\tPZ\n");

        printf("\x1b[91m timestep: %e\t%.10e\t%.10e\t%.10e\t%.10e\x1b[0m\n", idstep * dt, kbt, p[0], p[1], p[2]);
        fprintf(f, "%e\t%.10e\t%.10e\t%.10e\t%.10e\n", idstep * dt, kbt, p[0], p[1], p[2]);

        fclose(f);
    }
}

inline size_t hash_string(const char *buf)
{
    size_t result = 0;
    while( *buf != 0 ) {
        result = result * 31 + *buf++;
    }

    return result;
}


LocalComm::LocalComm()
{
    local_comm = MPI_COMM_NULL;
    local_rank = 0;
    local_nranks = 1;
}

void LocalComm::initialize(MPI_Comm _active_comm)
{
    active_comm = _active_comm;
    MPI_Comm_rank(active_comm, &rank);
    MPI_Comm_size(active_comm, &nranks);

    local_comm = active_comm;

    MPI_Get_processor_name(name, &len);
    size_t id = hash_string(name);

    MPI_Comm_split(active_comm, id, rank, &local_comm) ;

    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_nranks);
}

