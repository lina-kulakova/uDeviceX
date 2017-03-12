namespace sim {
  int s_n;
  Particle *s_pp; /* Solvent */
  Force    *s_ff;

  int r_n, r_nc, r_nt = RBCnt, r_nv = RBCnv;
  Particle *r_pp; /* RBC */
  Force    *r_ff;
  int      r_faces[MAX_FACE_NUM];
  float    *r_host_av;

  float4  *s_zip0; /* "zipped" version of Solvent array */
  ushort4 *s_zip1;

  Particle *s_pp0; /* Solvent (temporal buffer) */

  CellLists *cells, *wall_cells;

  H5PartDump  *s_dump; /* solvent dump */
  H5FieldDump *f_dump; /* field dump */

  Particle      s_pp_hst[MAX_PART_NUM]; /* solvent on host */
  Particle      r_pp_hst[MAX_PART_NUM]; /* RBC on host */
  Particle      w_pp_hst[MAX_PART_NUM]; /* wall particles
					   on host */
  Particle      sr_pp[MAX_PART_NUM];    /* solvent + RBC on host */

  int w_n; /* Wall particles */
  float4 *w_pp4;  /* (sic) */

  Particle* w_pp;

  int    *w_key;
  int     w_key_hst[MAX_PART_NUM];

  Logistic::KISS* rnd;

#ifdef GWRP
float rbc_xx[MAX_PART_NUM], rbc_yy[MAX_PART_NUM], rbc_zz[MAX_PART_NUM];
float sol_xx[MAX_PART_NUM], sol_yy[MAX_PART_NUM], sol_zz[MAX_PART_NUM];
int   iotags[MAX_PART_NUM];
#endif

}
