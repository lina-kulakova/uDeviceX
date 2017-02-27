namespace rdstr {
  int n_bulk;
  Particle* bulk;
  
  PinnedHostBuffer1<Particle> *rbuf[27], *sbuf[27]; /* send and recive
						       buffers */
  
  PinnedHostBuffer2<float3> *llo, *hhi; /* extents of RBCs */

  MPI_Comm ccom; /* Cartesian communicator */
  MPI_Request sendcntreq[26];
  int co[3]; /* coordinates of this processor */
  
  std::vector<MPI_Request> sendreq, recvreq, recvcntreq;
  int myrank, periods[3];
  int rnk_ne[27]; /* rank      of the neighbor */
  int ank_ne[27]; /* anti-rank of the neighbor */
  int recv_cnts[27];

  int ncome, nstay;

  DeviceBuffer<      float*> *_ddst;
  DeviceBuffer<const float*> *_dsrc;
}
