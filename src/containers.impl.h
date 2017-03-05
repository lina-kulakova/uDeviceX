namespace Cont {
std::vector<Geom> setup_read(const char *path2ic) {
  std::vector<Geom> tt;
  if (m::rank != 0) return tt;

  FILE *f = fopen(path2ic, "r");
  printf("READING FROM: <%s>\n", path2ic);

  while (true) {
    Geom t;
    float *mat = t.mat;
    for (int i = 0; i < 4*4; i++) if (fscanf(f, "%f", &mat[i]) != 1) goto done;
    tt.push_back(t);
  }
 done:
  fclose(f);
  printf("Reading %d CELLs from...<%s>\n", (int)tt.size(), path2ic);
  return tt;
}

void setup_bcast(std::vector<Geom> *tt)  {
  int n = tt->size(), sz = 1, root = 0;
  MC(MPI_Bcast(&n, sz,   MPI_INT, root, m::cart));
  tt->resize(n);
  sz = n*sizeof(Geom)/sizeof(float);
  Geom* D = &(tt->front());
  MC(MPI_Bcast( D, sz, MPI_FLOAT, root, m::cart));
}

int setup_select(Particle* pp, int nv,
		 std::vector<Geom> *tt, float *orig_xyzuvw) {
  float rr[3*MAX_VERT_NUM]; /* rbc vertices from the file */
  const char* fn = "rbc.off";
  off::f2vert(fn, rr);

  Particle pp_hst[MAX_PART_NUM];

  int c, mi[3], L[3] = {XS, YS, ZS};
  for (c = 0; c < 3; ++c) mi[c] = (m::coords[c] + 0.5) * L[c];
  int nc = 0;
  for (int i = 0; i < tt->size(); i++) {
    Geom t = (*tt)[i];
    float *mat = t.mat, com[3];
    for (c = 0; c < 3; ++c) {
      com[c] = mat[4*c + 3] - mi[c]; /* to local coordinates */
      if (2*com[c] < -L[c] || 2*com[c] > L[c]) goto next;
    }
    for (c = 0; c < 3; ++c) mat[4*c + 3] = com[c];
    rbc::initialize((float*)(pp + nv * i), mat, orig_xyzuvw);
    //    rot(&pp[nv*i], mat, rr);
    nc ++;
  next: ;
  }
  return nc;
}

int setup(Particle* pp, int nv, const char *path2ic, float *orig_xyzuvw) {
  std::vector<Geom> tt = setup_read(path2ic);
  setup_bcast(&tt); /* MPI */
  int nc = setup_select(pp, nv, &tt, orig_xyzuvw); /* cells for this subdomain */
  return nc;
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

void rbc_dump(int nc, Particle *p, int* triplets,
	      int nv, int nt, int id) {
    const char *format4ply = "ply/rbcs-%05d.ply";
    char buf[200];
    sprintf(buf, format4ply, id);
    if (m::rank == 0) mkdir("ply", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    ply_dump(buf, triplets, nc, nt, p, nv);
}

}
