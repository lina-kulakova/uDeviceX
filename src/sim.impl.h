namespace sim {

static void distr_s() {
  sdstr::pack(s_pp, s_n);
  sdstr::send();
  sdstr::bulk(s_n, cells->start, cells->count);
  s_n = sdstr::recv_count();
  sdstr::recv_unpack(s_pp0, s_zip0, s_zip1, s_n, cells->start, cells->count);
  std::swap(s_pp, s_pp0);
}

static void distr_r() {
  rdstr::extent(r_pp, r_nc, r_nv);
  rdstr::pack_sendcnt(r_pp, r_nc, r_nv);
  r_nc = rdstr::post(r_nv); r_n = r_nc * r_nv;
  rdstr::unpack(r_pp, r_nc, r_nv);
}

void remove_bodies_from_wall() {
  if (!rbcs) return;
  if (!r_nc) return;
  DeviceBuffer<int> marks(r_n);
  k_wall::fill_keys<<<k_cnf(r_n)>>>(r_pp, r_n, marks.D);

  std::vector<int> tmp(marks.S);
  CC(cudaMemcpy(tmp.data(), marks.D, sizeof(int) * marks.S, D2H));
  std::vector<int> tokill;
  for (int i = 0; i < r_nc; ++i) {
    bool valid = true;
    for (int j = 0; j < r_nv && valid; ++j)
      valid &= 0 == tmp[j + r_nv * i];
    if (!valid) tokill.push_back(i);
  }

  r_nc = Cont::rbc_remove(r_pp, r_nv, r_nc, &tokill.front(), tokill.size());
  r_n = r_nc * r_nv;
}

static void update_helper_arrays() {
  CC(cudaFuncSetCacheConfig(k_sim::make_texture, cudaFuncCachePreferShared));
  k_sim::make_texture<<<(s_n + 1023) / 1024, 1024, 1024 * 6 * sizeof(float)>>>
    (s_zip0, s_zip1, (float*)s_pp, s_n);
}

void create_walls() {
  dSync();
  s_n = wall::init(s_pp, s_n); /* number of survived particles */
  wall_created = true;

  k_sim::clear_velocity<<<k_cnf(s_n)>>>(s_pp, s_n);
  cells->build(s_pp, s_n, NULL, NULL);
  update_helper_arrays();
  remove_bodies_from_wall();
}

void forces_rbc() {
  if (rbcs) rbc::forces_nohost(r_nc, (float*)r_pp, (float*)r_ff,
			       r_host_av);
}

void forces_dpd() {
  DPD::pack(s_pp, s_n, cells->start, cells->count);
  DPD::local_interactions(s_pp, s_zip0, s_zip1,
			  s_n, s_ff, cells->start,
			  cells->count);
  DPD::post(s_pp, s_n);
  DPD::recv();
  DPD::remote_interactions(s_n, s_ff);
}

void clear_forces() {
  Cont::clear_forces(s_ff, s_n);
  if (rbcs) Cont::clear_forces(r_ff, r_n);
}

void forces_wall() {
  if (rbcs && wall_created) wall::interactions(r_pp, r_n, r_ff);
  if (wall_created)         wall::interactions(s_pp, s_n, s_ff);
}

void forces_cnt(std::vector<ParticlesWrap> *w_r) {
  if (contactforces) {
    cnt::build_cells(*w_r);
    cnt::bulk(*w_r);
  }
}

void forces_fsi(SolventWrap *w_s, std::vector<ParticlesWrap> *w_r) {
  fsi::bind_solvent(*w_s);
  fsi::bulk(*w_r);
}

void forces() {
  SolventWrap w_s(s_pp, s_n, s_ff, cells->start, cells->count);
  std::vector<ParticlesWrap> w_r;
  if (rbcs) w_r.push_back(ParticlesWrap(r_pp, r_n, r_ff));

  clear_forces();

  forces_dpd();
  forces_wall();
  forces_rbc();

  forces_cnt(&w_r);
  forces_fsi(&w_s, &w_r);

  rex::bind_solutes(w_r);
  rex::pack_p();
  rex::post_p();
  rex::recv_p();

  rex::halo(); /* fsi::halo(); cnt::halo() */

  rex::post_f();
  rex::recv_f();
}

void in_out() {
#ifdef GWRP
#include "sim.hack.h"
#endif
}

void dev2hst() { /* device to host  data transfer */
  CC(cudaMemcpyAsync(sr_pp, s_pp,
		     sizeof(Particle) * s_n, D2H, 0));
  if (rbcs)
    CC(cudaMemcpyAsync(&sr_pp[s_n], r_pp,
		       sizeof(Particle) * r_n, D2H, 0));
}

void dump_part() {
  if (!hdf5part_dumps) return;
  dev2hst(); /* TODO: do not need `r' */
  int n = s_n + r_n;
  dump_part_solvent->dump(sr_pp, n);
}

void dump_rbcs() {
  if (!rbcs) return;
  static int id = 0;
  dev2hst();  /* TODO: do not need `s' */
  Cont::rbc_dump(r_nc, &sr_pp[s_n], r_faces, r_n, r_nv, r_nt, id++);
}

void dump_grid() {
  if (!hdf5field_dumps) return;
  dev2hst();  /* TODO: do not need `r' */
  dump_field->dump(sr_pp, s_n);
}

void diag(int it) {
  int n = s_n + r_n; dev2hst();
  diagnostics(sr_pp, n, it);
}

void update() {
  k_sim::update<<<k_cnf(s_n)>>>
    (false, (float2*)s_pp, (float*)s_ff, s_n, driving_force);
  if (rbcs && r_n) k_sim::update<<<k_cnf(r_n)>>>
		     (true, (float2*)r_pp, (float*)r_ff, r_n, driving_force);
}

void bounce() {
  if (!wall_created) return;
  wall::bounce(s_pp, s_n);
  if (rbcs) wall::bounce(r_pp, r_n);
}

void init() {
  CC(cudaMalloc(&r_orig_xyzuvw, RBCnv * 6 * sizeof(float)));
  CC(cudaMalloc(&r_host_av, MAX_CELLS_NUM));
  
  rbc::setup(r_faces, r_orig_xyzuvw);
  rdstr::init();
  DPD::init();
  fsi::init();
  rex::init();
  cnt::init();
  if (hdf5part_dumps)
    dump_part_solvent = new H5PartDump("s.h5part");

  cells   = new CellLists(XS, YS, ZS);
  mpDeviceMalloc(&s_zip0); mpDeviceMalloc(&s_zip1);

  if (rbcs)
      mpDeviceMalloc(&r_pp); mpDeviceMalloc(&r_ff);

  wall::trunk = new Logistic::KISS;
  sdstr::init();
  mpDeviceMalloc(&s_pp); mpDeviceMalloc(&s_pp0);
  mpDeviceMalloc(&s_ff);
  mpDeviceMalloc(&r_ff); mpDeviceMalloc(&r_ff);

  s_n = ic::gen(s_pp_hst);
  CC(cudaMemcpy(s_pp, s_pp_hst, sizeof(Particle) * s_n, H2D));
  cells->build(s_pp, s_n, NULL, NULL);
  update_helper_arrays();

  if (rbcs) {
    r_nc = Cont::setup(r_pp, r_nv, "rbcs-ic.txt", r_orig_xyzuvw);
    r_n = r_nc * r_nv;
#ifdef GWRP
    iotags_init(r_nv, r_nt, r_faces);
    iotags_domain(0, 0, 0,
		  XS, YS, ZS,
		  m::periods[0], m::periods[1], m::periods[0]);
#endif
  }

  dump_field = new H5FieldDump;
  MC(MPI_Barrier(m::cart));
}

void dumps_diags(int it) {
  if (it % steps_per_dump == 0)     in_out();
  if (it % steps_per_dump == 0)     dump_rbcs();
  if (it % steps_per_dump == 0)     dump_part();
  if (it % steps_per_hdf5dump == 0) dump_grid();
  if (it % steps_per_dump == 0)     diag(it);
}

void run() {
  int nsteps = (int)(tend / dt);
  if (m::rank == 0 && !walls) printf("will take %ld steps\n", nsteps);
  if (!walls && pushtheflow) driving_force = hydrostatic_a;
  int it;
  for (it = 0; it < nsteps; ++it) {
    if (walls && it == wall_creation_stepid) {
      create_walls();
      if (rbcs && r_n) k_sim::clear_velocity<<<k_cnf(r_n)>>>(r_pp, r_n);
      if (pushtheflow) driving_force = hydrostatic_a;
    }
    distr_s();
    if (rbcs) distr_r();
    forces();
    dumps_diags(it);
    update();
    bounce();
  }
}

void close() {
  delete dump_field;
  delete dump_part_solvent;
  sdstr::redist_part_close();

  cnt::close();
  delete cells;
  rex::close();
  fsi::close();
  DPD::close();
  rdstr::close();

  CC(cudaFree(s_zip0));
  CC(cudaFree(s_zip1));

  CC(cudaFree(r_orig_xyzuvw));
  CC(cudaFree(r_host_av));

  delete wall::trunk;
  CC(cudaFree(r_pp )); CC(cudaFree(r_ff ));
  CC(cudaFree(s_pp )); CC(cudaFree(s_ff ));
  CC(cudaFree(s_pp0));
}
}