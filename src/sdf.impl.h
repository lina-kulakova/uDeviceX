namespace sdf {
  void setup() {
    k_sdf::texSDF.normalized = 0;
    k_sdf::texSDF.filterMode = cudaFilterModePoint;
    k_sdf::texSDF.mipmapFilterMode = cudaFilterModePoint;
    k_sdf::texSDF.addressMode[0] = cudaAddressModeWrap;
    k_sdf::texSDF.addressMode[1] = cudaAddressModeWrap;
    k_sdf::texSDF.addressMode[2] = cudaAddressModeWrap;
  }
  
  void init() {
    float grid_data[MAX_SUBDOMAIN_VOLUME];
    float *field = new float[XTEXTURESIZE * YTEXTURESIZE * ZTEXTURESIZE];
    int N[3];
    float extent[3];
    field::ini("sdf.dat", N, extent, grid_data);
    int L[3] = {XS, YS, ZS};
    int MARGIN[3] = {XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL};
    int TEXTURESIZE[3] = {XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE};
    if (m::rank == 0) printf("sampling the geometry file...\n");
    {
      float start[3], spacing[3];
      for (int c = 0; c < 3; ++c) {
	start[c] = N[c] * (m::coords[c] * L[c] - MARGIN[c]) /
	  (float)(m::dims[c] * L[c]);
	spacing[c] = N[c] * (L[c] + 2 * MARGIN[c]) /
	  (float)(m::dims[c] * L[c]) / (float)TEXTURESIZE[c];
      }
      float ampl = (XS /*+ 2 * XMARGIN_WALL*/) /
	(extent[0] / m::dims[0]);
      field::sample(start, spacing, TEXTURESIZE, N, ampl, grid_data,
		    field);
    }

    if (hdf5field_dumps) field::dump(N, extent, grid_data);
      
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    CC(cudaMalloc3DArray
       (&arrSDF, &fmt, make_cudaExtent(XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE)));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr
      ((void *)field, XTEXTURESIZE * sizeof(float), XTEXTURESIZE, YTEXTURESIZE);

    copyParams.dstArray = arrSDF;
    copyParams.extent = make_cudaExtent(XTEXTURESIZE, YTEXTURESIZE, ZTEXTURESIZE);
    copyParams.kind = H2D;
    CC(cudaMemcpy3D(&copyParams));
    delete[] field;
    
    setup();
    CC(cudaBindTextureToArray(k_sdf::texSDF, arrSDF, fmt));
  }

  void bulk_wall(Particle *s_pp, int* s_n, Particle *w_pp_hst, int *w_n,
		 int* w_key, int* w_key_hst) {
    /* sort particle into remaining in solvent and turning into wall */
    int n = *s_n;

    k_sdf::fill_keys<<<k_cnf(n)>>>(s_pp, n, w_key);
    CC(cudaMemcpy(w_key_hst, w_key, sizeof(int)*n, D2H));

    int k;
    int ia = 0, ib = 0, iw = 0; /* all, bulk, wall particles */
    for (/* */ ; ia < n; ia++) {
      k = w_key_hst[ia];
      if      (k == W_BULK)
	CC(cudaMemcpy(    &s_pp[ib++], &s_pp[ia], sizeof(Particle), D2D));
      else if (k == W_WALL)
	CC(cudaMemcpy(&w_pp_hst[iw++], &s_pp[ia], sizeof(Particle), D2H));
    }
    *s_n = ib; *w_n = iw;
  }

  void close() {
    CC(cudaUnbindTexture(k_sdf::texSDF));
    CC(cudaFreeArray(arrSDF));
  }
}
