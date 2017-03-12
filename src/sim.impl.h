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
  k_sdf::fill_keys<<<k_cnf(r_n)>>>(r_pp, r_n, marks.D);

  std::vector<int> tmp(marks.S);
  CC(cudaMemcpy(tmp.data(), marks.D, sizeof(int) * marks.S, D2H));
  std::vector<int> tokill;
  for (int i = 0; i < r_nc; ++i) {
    bool valid = true;
    for (int j = 0; j < r_nv && valid; ++j)
      valid &= (tmp[j + r_nv * i] == W_BULK);
    if (!valid) tokill.push_back(i);
  }

  r_nc = Cont::rbc_remove(r_pp, r_nv, r_nc, &tokill.front(), tokill.size());
  r_n = r_nc * r_nv;
}

static void update_helper_arrays() {
  k_sim::make_texture<<<(s_n + 1023) / 1024, 1024, 1024 * 6 * sizeof(float)>>>
    (s_zip0, s_zip1, (float*)s_pp, s_n);
}

void create_wall() {
  dSync();
  sdf::init();

  sdf::bulk_wall(s_pp, /*o*/ &s_n, w_pp_hst, &w_n, /*s*/ w_key, w_key_hst);

  if (hdf5part_dumps) {
    H5PartDump w_dump("w.h5part"); /* wall dump */
    w_dump.dump(w_pp_hst, w_n);
  }

  wall::exch(w_pp_hst, &w_n);
  if (w_n) {
    CC(cudaMemcpy(w_pp, w_pp_hst, sizeof(Particle)*w_n, H2D));
    wall_cells->build(w_pp, w_n, 0);
    k_wall::strip<<<k_cnf(w_n)>>>(w_pp, w_n, w_pp4);
  }

  k_sim::clear_velocity<<<k_cnf(s_n)>>>(s_pp, s_n);
  cells->build(s_pp, s_n, NULL, NULL);
  update_helper_arrays();
  remove_bodies_from_wall();
}

void forces_rbc() {
  if (rbcs) rbc::forces(r_nc, r_pp, r_ff, r_host_av);
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

void clear_forces(Force* ff, int n) {
  CC(cudaMemsetAsync(ff, 0, sizeof(Force) * n));
}

void forces_wall() {
  if (rbcs) wall::interactions(r_pp, w_pp4, r_n, w_n, wall_cells, rnd, r_ff);
  wall::interactions(s_pp, w_pp4, s_n, w_n, wall_cells, rnd, s_ff);
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

void forces(bool wall_created) {
  SolventWrap w_s(s_pp, s_n, s_ff, cells->start, cells->count);
  std::vector<ParticlesWrap> w_r;
  if (rbcs) w_r.push_back(ParticlesWrap(r_pp, r_n, r_ff));

  clear_forces(s_ff, s_n);
  if (rbcs) clear_forces(r_ff, r_n);

  forces_dpd();
  if (wall_created) forces_wall();
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
  s_dump->dump(sr_pp, n);
}

void dump_rbcs() {
  if (!rbcs) return;
  static int id = 0;
  dev2hst();  /* TODO: do not need `s' */
  Cont::rbc_dump(r_nc, &sr_pp[s_n], r_faces, r_nv, r_nt, id++);
}

void dump_grid() {
  if (!hdf5field_dumps) return;
  dev2hst();  /* TODO: do not need `r' */
  f_dump->dump(sr_pp, s_n);
}

void diag(int it) {
  int n = s_n + r_n; dev2hst();
  diagnostics(sr_pp, n, it);
}

void body_force(float driving_force) {
  k_sim::body_force<<<k_cnf(s_n)>>> (false, s_pp, s_ff, s_n, driving_force);

  if (!rbcs || !r_n) return;
  k_sim::body_force<<<k_cnf(r_n)>>> (true, r_pp, r_ff, r_n, driving_force);
}

void update() {
  if (s_n)         k_sim::update<<<k_cnf(s_n)>>>(false, s_pp, s_ff, s_n);
  if (rbcs && r_n) k_sim::update<<<k_cnf(r_n)>>>( true, r_pp, r_ff, r_n);

}

void bounce() {
  if (s_n)         k_sdf::bounce<<<k_cnf(s_n)>>>((float2*)s_pp, s_n);
  if (rbcs && r_n) k_sdf::bounce<<<k_cnf(r_n)>>>((float2*)r_pp, r_n);
}

void init() {
  CC(cudaMalloc(&r_host_av, MAX_CELLS_NUM));

  rbc::setup(r_faces);
  rdstr::init();
  DPD::init();
  fsi::init();
  rex::init();
  cnt::init();
  if (hdf5part_dumps)
    s_dump = new H5PartDump("s.h5part");

  cells      = new CellLists(XS, YS, ZS);
  wall_cells = new CellLists(XS + 2 * XMARGIN_WALL,
			     YS + 2 * YMARGIN_WALL,
			     ZS + 2 * ZMARGIN_WALL);
  mpDeviceMalloc(&s_zip0); mpDeviceMalloc(&s_zip1);

  if (rbcs)
      mpDeviceMalloc(&r_pp); mpDeviceMalloc(&r_ff);

  rnd = new Logistic::KISS;
  sdstr::init();
  mpDeviceMalloc(&s_pp); mpDeviceMalloc(&s_pp0);
  mpDeviceMalloc(&s_ff);
  mpDeviceMalloc(&r_ff); mpDeviceMalloc(&r_ff);

  mpDeviceMalloc(&w_pp);
  mpDeviceMalloc(&w_pp4); mpDeviceMalloc(&w_key);

  s_n = ic::gen(s_pp_hst);
  CC(cudaMemcpy(s_pp, s_pp_hst, sizeof(Particle) * s_n, H2D));
  cells->build(s_pp, s_n, NULL, NULL);
  update_helper_arrays();

  if (rbcs) {
    r_nc = Cont::setup(r_pp, r_nv, /* storage */ r_pp_hst);

    r_n = r_nc * r_nv;
#ifdef GWRP
    iotags_init(r_nv, r_nt, r_faces);
    iotags_domain(0, 0, 0,
		  XS, YS, ZS,
		  m::periods[0], m::periods[1], m::periods[0]);
#endif
  }

  f_dump = new H5FieldDump;
  MC(MPI_Barrier(m::cart));
}

void dumps_diags(int it) {
  if (it % steps_per_dump == 0)     in_out();
  if (it % steps_per_dump == 0)     dump_rbcs();
  if (it % steps_per_dump == 0)     dump_part();
  if (it % steps_per_hdf5dump == 0) dump_grid();
  if (it % steps_per_dump == 0)     diag(it);
}

void run0(bool wall_created, float driving_force, int it) {
  distr_s();
  if (rbcs) distr_r();
  forces(wall_created);
  dumps_diags(it);
  body_force(driving_force);
  update();
  if (wall_created) bounce();
}

void run_nowall(int nsteps) {
  bool wall_created = false;
  float driving_force = pushtheflow ? hydrostatic_a : 0;
  for (int it = 0; it < nsteps; ++it)
    run0(wall_created, driving_force, it);
}

void run_wall(int nsteps) {
  bool wall_created = false;
  float driving_force = 0;
  int it = 0;
  for (/* */; it < wall_creation_stepid; ++it)
    run0(wall_created, driving_force, it);

  create_wall(); wall_created = true;
  if (rbcs && r_n) k_sim::clear_velocity<<<k_cnf(r_n)>>>(r_pp, r_n);
  driving_force = pushtheflow ? hydrostatic_a : 0;

  for (/* */; it < nsteps; ++it)
    run0(wall_created, driving_force, it);
}

void run() {
  int nsteps = (int)(tend / dt);
  if (walls) run_wall(nsteps); else run_nowall(nsteps);
}

void close() {
  delete s_dump;
  delete f_dump;

  sdstr::redist_part_close();

  cnt::close();
  delete cells;
  delete wall_cells;
  rex::close();
  fsi::close();
  DPD::close();
  rdstr::close();

  CC(cudaFree(s_zip0));
  CC(cudaFree(s_zip1));
  CC(cudaFree(r_host_av));

  delete rnd;

  CC(cudaFree(r_pp )); CC(cudaFree(r_ff ));
  CC(cudaFree(s_pp )); CC(cudaFree(s_ff ));
  CC(cudaFree(s_pp0));

  CC(cudaFree(w_pp4)); CC(cudaFree(w_key));
  CC(cudaFree(w_pp));
}
}
