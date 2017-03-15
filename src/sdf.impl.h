namespace sdf {
  void setup() {
    k_sdf::texSDF.normalized = 0;
    k_sdf::texSDF.filterMode = cudaFilterModePoint;
    k_sdf::texSDF.mipmapFilterMode = cudaFilterModePoint;
    k_sdf::texSDF.addressMode[0] = cudaAddressModeWrap;
    k_sdf::texSDF.addressMode[1] = cudaAddressModeWrap;
    k_sdf::texSDF.addressMode[2] = cudaAddressModeWrap;
  }

  /* go from spacing for the input grid in sdf file to spacing for the
     field used in simulation (input to field) */  
  void i2f(int * i_N, float* i_extent, int *TEXTURESIZE, /**/
	   float* start, float* spacing, float* ampl) {
    int L[3] = {XS, YS, ZS};
    int MARGIN[3] = {XMARGIN_WALL, YMARGIN_WALL, ZMARGIN_WALL};
    for (int c = 0; c < 3; ++c) {
      start[c] = i_N[c] * (m::coords[c] * L[c] - MARGIN[c]) /
	(float)(m::dims[c] * L[c]);
      spacing[c] = i_N[c] * (L[c] + 2 * MARGIN[c]) /
	(float)(m::dims[c] * L[c]) / (float)TEXTURESIZE[c];
    }
    *ampl = XS / (i_extent[0] / m::dims[0]);
  }
  
  void init(/**/ float* field) {
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
