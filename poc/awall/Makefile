all: libawall_hst.a libawall_dev.a main

NVCC_DIR = /usr/local/cuda-8.0
NVCC_BIN = $(NVCC_DIR)/bin/nvcc
NVCC = $(NVCC_BIN)
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35
NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL)

libawall_dev.a: awall_dev.o; ar rv $@ $<
libawall_hst.a: awall_hst.o; ar rv $@ $<

awall_dev.o: awall.cpp; $(NVCC) -x cu $(NVCCFLAGS) -dc $< -c -o $@
awall_hst.o: awall.cpp; $(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -c -o $@

# a test client for the library
LDLIBS+=-lm
main: main.o libawall_hst.a
main.o: main.cpp

clean:; -rm awall.o libawall_hst.a libawall_dev.a awall_dev.o awall_hst.o
