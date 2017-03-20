all: libhd_hst.a libhd_dev.a


NVCC_DIR = /usr/local/cuda-8.0
NVCC_BIN = $(NVCC_DIR)/bin/nvcc
NVCC = $(NVCC_BIN)
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35
NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL)

libhd_dev.a: hd_dev.o; ar rv $@ $<
libhd_hst.a: hd_hst.o; ar rv $@ $<

hd_dev.o: hd.cpp; $(NVCC) -x cu $(NVCCFLAGS) -dc $< -c -o $@
hd_hst.o: hd.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

clean:; -rm hd.o libhd_hst.a libhd_dev.a hd_dev.o hd_hst.o
