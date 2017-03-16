namespace wall {
  void exch(Particle *w_pp_hst, int* w_n) {
      int dstranks[26], remsizes[26], recv_tags[26];
      for (int i = 0; i < 26; ++i) {
	int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};

	recv_tags[i] =
	  (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

	int coordsneighbor[3];
	for (int c = 0; c < 3; ++c) coordsneighbor[c] = m::coords[c] + d[c];
	MC(MPI_Cart_rank(m::cart, coordsneighbor, dstranks + i));
      }

      // send local counts - receive remote counts
      {
	for (int i = 0; i < 26; ++i) remsizes[i] = -1;

	MPI_Request reqrecv[26];
	for (int i = 0; i < 26; ++i)
	  MC(MPI_Irecv(remsizes + i, 1, MPI_INTEGER, dstranks[i],
		       123 + recv_tags[i], m::cart, reqrecv + i));

	MPI_Request reqsend[26];
	for (int i = 0; i < 26; ++i)
	  MC(MPI_Isend(w_n, 1, MPI_INTEGER, dstranks[i], 123 + i,
		       m::cart, reqsend + i));
	MPI_Status statuses[26];
	MC(MPI_Waitall(26, reqrecv, statuses));
	MC(MPI_Waitall(26, reqsend, statuses));
      }

      std::vector<Particle> remote[26];
      // send local data - receive remote data
      {
	for (int i = 0; i < 26; ++i) remote[i].resize(remsizes[i]);

	MPI_Request reqrecv[26];
	for (int i = 0; i < 26; ++i)
	  MC(MPI_Irecv(remote[i].data(), remote[i].size() * 6, MPI_FLOAT,
		       dstranks[i], 321 + recv_tags[i], m::cart,
		       reqrecv + i));
	MPI_Request reqsend[26];
	for (int i = 0; i < 26; ++i)
	  MC(MPI_Isend(w_pp_hst, (*w_n) * 6, MPI_FLOAT,
		       dstranks[i], 321 + i, m::cart, reqsend + i));

	MPI_Status statuses[26];
	MC(MPI_Waitall(26, reqrecv, statuses));
	MC(MPI_Waitall(26, reqsend, statuses));
      }
      MPI_Barrier(m::cart);

      int L[3] = {XS, YS, ZS};
      int MARGIN[3] = {XWM, YWM, ZWM};
      float lo[3], hi[3];
      int c, i, j;
      for (c = 0; c < 3; c ++) {
	lo[c] = -0.5*L[c] - MARGIN[c];
	hi[c] =  0.5*L[c] + MARGIN[c];
      }

      for (i = 0; i < 26; ++i) {
	int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};
	for (j = 0; j < remote[i].size(); ++j) {
	  Particle p = remote[i][j];
	  for (c = 0; c < 3; ++c) {
	    p.r[c] += d[c] * L[c];
	    if (p.r[c] < lo[c] || p.r[c] >= hi[c]) goto next;
	  }
	  w_pp_hst[(*w_n)++] = p;
	next: ;
	}
      }
  } /* end of ini */

  void init_textrue() {
    setup_texture(k_wall::texWallParticles, float4);
    setup_texture(k_wall::texWallCellStart, int);
    setup_texture(k_wall::texWallCellCount, int);
  }

  void interactions(Particle *s_pp, float4* w_pp4, int s_n, int w_n,
		    CellLists* cells, Logistic::KISS* rnd,
		    Force *ff) {
    init_textrue();
    if (s_n && w_n) {
      size_t offset;
      CC(cudaBindTexture(&offset,
			 &k_wall::texWallParticles, w_pp4,
			 &k_wall::texWallParticles.channelDesc,
			 sizeof(float4) * w_n));

      CC(cudaBindTexture(&offset,
			 &k_wall::texWallCellStart, cells->start,
			 &k_wall::texWallCellStart.channelDesc,
			 sizeof(int) * cells->ncells));

      CC(cudaBindTexture(&offset,
			 &k_wall::texWallCellCount, cells->count,
			 &k_wall::texWallCellCount.channelDesc,
			 sizeof(int) * cells->ncells));

      k_wall::interactions_3tpp<<<k_cnf(3*s_n)>>>
	((float2*)s_pp, s_n, w_n, (float*)ff, rnd->get_float());

      CC(cudaUnbindTexture(k_wall::texWallParticles));
      CC(cudaUnbindTexture(k_wall::texWallCellStart));
      CC(cudaUnbindTexture(k_wall::texWallCellCount));
    }
  }
}
