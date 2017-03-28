# TEST: sim.t0
# cp .conf.2cy.h .conf.h
# (make clean && make -j) 2>/dev/null 1>/dev/null
# ./sim
# ../../tools/fround.awk -v tol=2 3d/01000.3D > sim.out.3D

# TEST: sim.t1
# cp .conf.pl.h  .conf.h
# (make clean && make -j) 2>/dev/null 1>/dev/null
# ./sim
# ../../tools/fround.awk -v tol=2 3d/01000.3D > sim.out.3D
