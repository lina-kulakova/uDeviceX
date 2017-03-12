include .cache.Makefile # local settings (see .cache.Makefile.* for examples)

NVCC     ?= nvcc
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35
OPT      ?= -O3 -use_fast_math -g

slevel   ?= -2

NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL)
LDFLAGS   += -L../cuda-dpd -L../last-bit
NVCCFLAGS += -I../cuda-dpd -I../last-bit -I../helper-math -I.
LIBS      += -lcuda-dpd -llast-bit -lcudart

ifeq ($(h5part),0)
NVCCFLAGS += -DNO_H5PART
endif

ifeq ($(h5),0)
NVCCFLAGS += -DNO_H5
endif

ifeq ($(gwrp),1) # geom-wrapper uses CGAL
NVCCFLAGS += -DGWRP
endif

NVCCFLAGS += -DVISCOSITY_S_LEVEL=$(slevel)
OBJS = 	common.o io.o main.o minmax.o m.o    bund.o glb.o dpd-forces.o

# I have some device code in OBJS. I have to link it separatly
udx: $(OBJS) libcuda-dpd liblast-bit
	$(NVCC) $(NVCCFLAGS) $(OPT) -dlink \
		-L../cuda-dpd -L../last-bit $(OBJS) $(LIBS) -o gpuCode.o
	$(LINK)  $(LDFLAGS) gpuCode.o $(OBJS) $(LIBS) -o $@

common.o:     common.cu;     $(NVCC) $(NVCCFLAGS) $(OPT)     $< -c -o $@
io.o:         io.cu;         $(NVCC) $(NVCCFLAGS) $(OPT)     $< -c -o $@
main.o:       main.cu;       $(NVCC) $(NVCCFLAGS) $(OPT)     $< -c -o $@
minmax.o:     minmax.cu;     $(NVCC) $(NVCCFLAGS) $(OPT)     $< -c -o $@
m.o:          m.cu;          $(NVCC) $(NVCCFLAGS) $(OPT)     $< -c -o $@

# some files must be compiled with `-dc'. See "Separate Compilation
# and Linking of CUDA C++ Device Code"
# http://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code
bund.o:       bund.cu;       $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
glb.o:        glb.cu;        $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
dpd-forces.o: dpd-forces.cu; $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
include    Makefile.dep # dependencies

libcuda-dpd:; make -C ../cuda-dpd \
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" OPT="$(OPT)"

liblast-bit:; make -C ../last-bit \
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" OPT="$(OPT)"

clean:  ; rm -f udx *.o
	make -C ../cuda-dpd clean
	make -C ../last-bit clean

.PHONY = clean libcuda-dpd liblast-bit
