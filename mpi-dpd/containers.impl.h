namespace ParticleKernels {
    __global__ void update(bool rbcflag, float2 * _pdata, float * _adata,
				   int nparticles, float _driving_force) {
	int warpid = threadIdx.x >> 5;
	int base = 32 * (warpid + 4 * blockIdx.x);
	int nsrc = min(32, nparticles - base);

	float2 *pdata = _pdata + 3 * base;
	float *adata = _adata + 3 * base;

	int laneid;
	asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));

	int nwords = 3 * nsrc;

	float2 s0, s1, s2;
	float ax, ay, az;

	if (laneid < nwords)
	{
	    s0 = __ldg(pdata + laneid);
	    ax = __ldg(adata + laneid);
	}

	if (laneid + 32 < nwords)
	{
	    s1 = __ldg(pdata + laneid + 32);
	    ay = __ldg(adata + laneid + 32);
	}

	if (laneid + 64 < nwords)
	{
	    s2 = __ldg(pdata + laneid + 64);
	    az = __ldg(adata + laneid + 64);
	}

	{
	    int srclane0 = (3 * laneid + 0) & 0x1f;
	    int srclane1 = (srclane0 + 1) & 0x1f;
	    int srclane2 = (srclane0 + 2) & 0x1f;

	    int start = laneid % 3;

	    {
		float t0 = __shfl(start == 0 ? s0.x : start == 1 ? s1.x : s2.x, srclane0);
		float t1 = __shfl(start == 0 ? s2.x : start == 1 ? s0.x : s1.x, srclane1);
		float t2 = __shfl(start == 0 ? s1.x : start == 1 ? s2.x : s0.x, srclane2);

		s0.x = t0;
		s1.x = t1;
		s2.x = t2;
	    }

	    {
		float t0 = __shfl(start == 0 ? s0.y : start == 1 ? s1.y : s2.y, srclane0);
		float t1 = __shfl(start == 0 ? s2.y : start == 1 ? s0.y : s1.y, srclane1);
		float t2 = __shfl(start == 0 ? s1.y : start == 1 ? s2.y : s0.y, srclane2);

		s0.y = t0;
		s1.y = t1;
		s2.y = t2;
	    }

	    {
		float t0 = __shfl(start == 0 ? ax : start == 1 ? ay : az, srclane0);
		float t1 = __shfl(start == 0 ? az : start == 1 ? ax : ay, srclane1);
		float t2 = __shfl(start == 0 ? ay : start == 1 ? az : ax, srclane2);

		ax = t0;
		ay = t1;
		az = t2;
	    }
	}

	int type; float driving_force, mass, vx = s1.y, y = s0.y;
	if      (rbcflag                             ) type = MEMB_TYPE;
	else if (!rbcflag &&  lastbit::get(vx)) type =  IN_TYPE;
	else if (!rbcflag && !lastbit::get(vx)) type = OUT_TYPE;
	mass          = (type == MEMB_TYPE) ? rbc_mass : 1;
	/* TODO: no driving force on "inner" particles */
	driving_force = (type ==   IN_TYPE) ? 0        : _driving_force;
	float y0 = glb::r0[1]; /* domain center in local coordinates */
	if (doublepoiseuille && y <= y0) driving_force *= -1;

	s1.y += (ax/mass + driving_force) * dt;
	s2.x += ay/mass * dt;
	s2.y += az/mass * dt;

	s0.x += s1.y * dt;
	s0.y += s2.x * dt;
	s1.x += s2.y * dt;

	{
	    int srclane0 = (32 * ((laneid) % 3) + laneid) / 3;
	    int srclane1 = (32 * ((laneid + 1) % 3) + laneid) / 3;
	    int srclane2 = (32 * ((laneid + 2) % 3) + laneid) / 3;

	    int start = laneid % 3;

	    {
		float t0 = __shfl(s0.x, srclane0);
		float t1 = __shfl(s2.x, srclane1);
		float t2 = __shfl(s1.x, srclane2);

		s0.x = start == 0 ? t0 : start == 1 ? t2 : t1;
		s1.x = start == 0 ? t1 : start == 1 ? t0 : t2;
		s2.x = start == 0 ? t2 : start == 1 ? t1 : t0;
	    }

	    {
		float t0 = __shfl(s0.y, srclane0);
		float t1 = __shfl(s2.y, srclane1);
		float t2 = __shfl(s1.y, srclane2);

		s0.y = start == 0 ? t0 : start == 1 ? t2 : t1;
		s1.y = start == 0 ? t1 : start == 1 ? t0 : t2;
		s2.y = start == 0 ? t2 : start == 1 ? t1 : t0;
	    }

	    {
		float t0 = __shfl(ax, srclane0);
		float t1 = __shfl(az, srclane1);
		float t2 = __shfl(ay, srclane2);

		ax = start == 0 ? t0 : start == 1 ? t2 : t1;
		ay = start == 0 ? t1 : start == 1 ? t0 : t2;
		az = start == 0 ? t2 : start == 1 ? t1 : t0;
	    }
	}

	if (laneid < nwords) {
	    lastbit::Preserver up1(pdata[laneid].y);
	    pdata[laneid] = s0;
	}

	if (laneid + 32 < nwords) {
	    lastbit::Preserver up1(pdata[laneid + 32].y);
	    pdata[laneid + 32] = s1;
	}

	if (laneid + 64 < nwords) {
	    lastbit::Preserver up1(pdata[laneid + 64].y);
	    pdata[laneid + 64] = s2;
	}
    }

    __global__ void clear_velocity(Particle *p, int n)  {
	int pid = threadIdx.x + blockDim.x * blockIdx.x;
	if (pid >= n) return;
	lastbit::Preserver up(p[pid].v[0]);
	for(int c = 0; c < 3; ++c) p[pid].v[c] = 0;
    }
} /* end of ParticleKernels */

namespace Cont {
void  update(Particle* pp, Force* ff, int n,
	     bool rbcflag, float driving_force) {
  if (!n) return;
  ParticleKernels::update<<<(n + 127) / 128, 128, 0>>>
    (rbcflag, (float2*)pp, (float*)ff, n, driving_force);
}

void clear_velocity(Particle* pp, int n) {
  if (n)
    ParticleKernels::clear_velocity<<<(n + 127) / 128, 128 >>>(pp, n);
}

void _initialize(float *device_pp, float (*transform)[4], float* orig_xyzuvw) {
  rbc::initialize(device_pp, transform, orig_xyzuvw);
}

int setup(Particle* pp, const char *path2ic, int nv, float* orig_xyzuvw) {
  std::vector<Geom> tt;
  if (m::rank == 0) {
    //read mated extent from file
    FILE *f = fopen(path2ic, "r");
    printf("READING FROM: <%s>\n", path2ic);

    while (true) {
      float tmp[19];
      Geom t;
      int c, rcode, ctr = 0;
      for(c = 0; c < 19; ++c)
	if (fscanf(f, "%f", tmp + c) != 1) goto done;
      
      t.com[0] = tmp[ctr++]; t.com[1] = tmp[ctr++]; t.com[2] = tmp[ctr++];
      for(c = 0; c < 16; ++c) t.mat[c / 4][c % 4] = tmp[ctr++];
      tt.push_back(t);
    }
done:
    fclose(f);
    printf("Instantiating %d CELLs from...<%s>\n", (int)tt.size(), path2ic);
  } /* end of m::rank == 0 */

  int tt_count = tt.size();
  MC(MPI_Bcast(&tt_count, 1, MPI_INT, 0, m::cart));

  tt.resize(tt_count);

  int nfloats_per_entry = sizeof(Geom) / sizeof(float);
  MC(MPI_Bcast(&tt.front(), nfloats_per_entry * tt_count, MPI_FLOAT, 0, m::cart));

  std::vector<Geom> good;
  int L[3] = { XS, YS, ZS };

  for(std::vector<Geom>::iterator it = tt.begin(); it != tt.end(); ++it) {
    bool inside = true;
    for(int c = 0; c < 3; ++c)
      inside &= it->com[c] >= m::coords[c] * L[c] && it->com[c] < (m::coords[c] + 1) * L[c];
    if (inside) {
      for(int c = 0; c < 3; ++c)
	it->mat[c][3] -= (m::coords[c] + 0.5) * L[c];
      good.push_back(*it);
    }
  }

  int gs = good.size();
  for(int i = 0; i < gs; ++i)
    _initialize((float *)(pp + nv * i), good[i].mat, orig_xyzuvw);
  return gs; /* number of cells */
}

int rbc_remove(Particle* pp, int nv, int nc, int *e, int ne) {
  /* remove RBCs with indexes in `e' */
  bool GO = false, STAY = true;
  int ie, i0, i1;
  std::vector<bool> m(nc, STAY);
  for (ie = 0; ie < ne; ie++) m[e[ie]] = GO;

  for (i0 = i1 = 0; i0 < nc; i0++)
    if (m[i0] == STAY)
      CC(cudaMemcpy(pp + nv * (i1++), pp + nv * i0,
		    sizeof(Particle) * nv, D2D));
  int nstay = i1;
  return nstay;
}

void clear_forces(Force* ff, int n) {
  CC(cudaMemsetAsync(ff, 0, sizeof(Force) * n));
}

void rbc_dump(int nc, Particle *p, int* triplets,
	      int n, int nv, int nt, int id) {
    const char *format4ply = "ply/rbcs-%05d.ply";
    char buf[200];
    sprintf(buf, format4ply, id);
    if (m::rank == 0) mkdir("ply", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    ply_dump(buf, triplets, nc, nt, p, nv);
}

}
