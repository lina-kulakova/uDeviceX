/*

Usage:

# TEST: iotags.t1
# export nb=498
# export xl=0 yl=-10 zl=-10 xh=10 yh=10 zh=10
# export pbcx=1 pbcy=1 pbcz=1
# ./iotags test_data/faces.bin test_data/rbc.bin test_data/solvent.bin        tags.out.bin

# TEST: iotags.t2
# export nb=498
# export xl=0 yl=-10 zl=-10 xh=10 yh=10 zh=10
# export pbcx=0 pbcy=0 pbcz=0
# ./iotags test_data/faces.bin test_data/rbc.bin test_data/solvent.bin        tags.out.bin

# TEST: iotags.t3
# export nb=498
# export xl=0 yl=0 zl=0 xh=32 yh=32 zh=32
# export pbcx=1 pbcy=1 pbcz=1
# ./iotags test_data/faces.two.bin test_data/rbc.two.bin test_data/solvent.two.bin     tags.out.bin

 */

#include <stdio.h>
#include <vector>
#include "geom-wrapper.h"
#include "rbc_utils.h"
#include "off.impl.h"

using std::vector;

#define MAX_FACE_NUM 1000
#define MAX_PART_NUM 1000
#define NV_PER_FACE 3

int faces[3*MAX_FACE_NUM];
static int   nvar; /* number of variables in one line of input file
		      x, y, z (3, default), x, y, z, vx, vy, vz (6) */

static int   pbcx, pbcy, pbcz; /* domain */
static float xl, yl, zl,   xh, yh, zh;

namespace rbc {
  long   nv; /* number of vertices */
  long   nf; /* number of faces */
  long   nb; /* number of vertices in one RBC (number of beads) */

  vector<float>  xx, yy, zz;
  vector<int>  ff1, ff2, ff3;

  FILE* fd; /* rbc file (not faces) */

  void read_faces0(FILE* fd) {
    auto sz = nf*(NV_PER_FACE+1);
    vector<int> fbuf(sz);
    ff1.resize(nf), ff2.resize(nf), ff3.resize(nf);
    safe_fread(fbuf.data(), sz, sizeof(int), fd);
    for (int ifa = 0, ib = 0; ifa < nf; ++ifa) {
      ib++; /* skip nvpf */
      ff1[ifa] = fbuf[ib++]; ff2[ifa] = fbuf[ib++]; ff3[ifa] = fbuf[ib++];
    }
  }

  void read_faces(const char* fn) {
    auto fd = safe_fopen(fn, "r");
    nf = gnp<int>(fd, NV_PER_FACE + 1);
    read_faces0(fd);
    fclose(fd);
  }

  void init_rbc_file(const char* fn) {
    fd = safe_fopen(fn, "r");
    nv = gnp<float>(fd, nvar);
  }

  void read_vertices() { /* read all RBCs vertices */
    auto sz = nv*nvar;
    vector<float> buf(sz);
    safe_fread(buf.data(), sz, sizeof(float), fd);
    xx.resize(nv); yy.resize(nv); zz.resize(nv);
    for (int iv = 0, ib = 0; iv < nv; iv++) {
      xx[iv] = buf[ib++]; yy[iv] = buf[ib++]; zz[iv] = buf[ib++];
      ib += nvar - 3; /* skip the rest of variables */
    }
  }

  void close_file() {fclose(fd);}
}

namespace sol {
  long nv; /* number of points */
  vector<float> xx,  yy,  zz;  /* shifted to the current RBC */

  FILE* fd;

  void read_vertices() {
    auto sz = nv * nvar;
    std::vector<float> buf(sz);
    safe_fread(buf.data(), sz, sizeof(float), fd);
    xx.resize(nv); yy.resize(nv); zz.resize(nv);
    for (int iv = 0, ib = 0; iv < nv; iv++) {
      xx[iv] = buf[ib++]; yy[iv] = buf[ib++]; zz[iv] = buf[ib++];
      ib += nvar - 3; /* skip the rest of variables */
    }
  }

  void read_file(const char* fn) {
      fd = safe_fopen(fn, "r");
      nv = gnp<float>(fd, nvar);
      read_vertices();
      fclose(fd);
  }
}

void init() {
  rbc::nb = env2d("nb");
  xl = env2f("xl"), yl = env2f("yl"), zl = env2f("zl");
  xh = env2f("xh"), yh = env2f("yh"), zh = env2f("zh");
  pbcx = env2d("pbcx"), pbcy = env2d("pbcy"), pbcz = env2d("pbcz");
  nvar = env2d_default("nvar", 3);
}


void write_iotags(const char* fn, vector<int> iotags) {
  fprintf(stderr, "(iotags) writing: %s\n", fn);
  FILE* fd = safe_fopen(fn, "w");
  fwrite(iotags.data(), sol::nv, sizeof(float), fd);
  fclose(fd);
}

int main(int argc, const char** argv) {
  int iarg = 1;

  /* [f]ile [n]ames */
  auto faces_fn = argv[iarg++], rbc_fn = argv[iarg++], \
    solvent_fn = argv[iarg++], tags_fn = argv[iarg++];

  init();
  sol::read_file(solvent_fn);
  rbc::read_faces(faces_fn);

  rbc::init_rbc_file(rbc_fn);

  iotags_domain(xl, yl, zl,    xh, yh, zh,    pbcx, pbcy, pbcz);
  int nv, nf;
  off::f2faces("test_data/rbc.off", faces, &nv, &nf);
  iotags_init(nv, nf, faces);
  
  rbc::read_vertices();
  vector<int> iotags(sol::nv, 0);
  iotags_all(rbc::nv, rbc::xx.data(), rbc::yy.data(), rbc::zz.data(),
	     sol::nv, sol::xx.data(), sol::yy.data(), sol::zz.data(),
	     iotags.data());
  rbc::close_file();
  write_iotags(tags_fn, iotags);
}
