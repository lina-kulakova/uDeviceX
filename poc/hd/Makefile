all: libhd_hst.a libhd_dev.a

libhd_hst.a: hd.o; ar rv $@ $<
hd.o: hd.cpp

NVCC_DIR = /usr/local/cuda-8.0
NVCC_BIN = $(NVCC_DIR)/bin/nvcc
NVCC = $(NVCC_BIN)
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35
NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL)

libhd_dev.a: hd2.o; ar rv $@ $<
hd2.o: hd2.cu; $(NVCC) $(NVCCFLAGS) -dc $< -c -o $@

clean:; -rm hd.o libhd_hst.a hd2.o libhd_dev.a 
