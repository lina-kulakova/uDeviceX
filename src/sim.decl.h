namespace sim {
  int s_n;
  Particle *s_pp; /* Solvent */
  Force    *s_ff;

  float4  *s_zip0; /* "zipped" version of Solvent array */
  ushort4 *s_zip1;

  Particle *s_pp0; /* Solvent (temporal buffer) */

  CellLists* cells;

  bool wall_created = false;
  float driving_force = 0;

  H5PartDump *dump_part_solvent;
  H5FieldDump *dump_field;

  Particle      s_pp_hst[MAX_PART_NUM]; /* solvent on host */
}
