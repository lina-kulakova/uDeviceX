NVCC_DIR = /usr/local/cuda-8.0
NVCC_BIN = $(NVCC_DIR)/bin/nvcc
NVCC = $(NVCC_BIN)
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35
NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL)
NVCCFLAGS += -I../hd
NVCC_LDFALGS = -L${NVCC_DIR}/lib64 -Wl,-rpath -Wl,${NVCC_DIR}/lib64

LOADLIBES += -L../hd
LIBS += -lhd_dev
LINK = g++

main: main.o
	$(NVCC) $(NVCCFLAGS) -dlink $(LOADLIBES)                           main.o $(LIBS)          -o gpuCode.o
	$(LINK)  $(LDFLAGS)         $(LOADLIBES) $(NVCC_LDFALGS) gpuCode.o main.o $(LIBS) -lcudart -o $@

main.o: main.cu; $(NVCC) $(NVCCFLAGS) -dc $< -c -o $@

clean:; -rm -f main.o main
