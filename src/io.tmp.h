namespace io { /* input-output */
  void wall_dump(int N[3], float extent[3], float* grid_data) {
    int c, L[3] = {XS, YS, ZS};
    float walldata[MAX_SUBDOMAIN_VOLUME];

    float rlo[3], dr[3], ampl;
    for (c = 0; c < 3; ++c) {
      rlo[c] = m::coords[c] * L[c] / (float)(m::dims[c] * L[c]) * N[c];
      dr[c] = N[c] / (float)(m::dims[c] * L[c]);
    }
    ampl = L[0] / (extent[0] / (float) m::dims[0]);
    field::sample(rlo, dr, L, N, ampl, grid_data, walldata);
    H5FieldDump dump;
    dump.dump_scalarfield(walldata, "wall");
  }
}
