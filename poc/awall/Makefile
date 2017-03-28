all: sim libawall_hst.a libawall_dev.a

NVCC_DIR = /usr/local/cuda-8.0
NVCC_BIN = $(NVCC_DIR)/bin/nvcc
NVCC = $(NVCC_BIN)
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35
NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL)
CXXFLAGS = -Wall -Wextra -O0 -g

libawall_dev.a: awall_dev.o; ar rv $@ $<
libawall_hst.a: awall_hst.o; ar rv $@ $<

awall.up.o    : awall.cpp awall.h 
awall_dev.o: awall.up.cpp; $(NVCC) -x cu $(NVCCFLAGS) -dc $< -c -o $@
awall_hst.o: awall.up.cpp; $(CXX) $(CXXFLAGS) $< -c -o $@

# a test client for the library
LDLIBS+=-lm
sim: sim.o libawall_hst.a
sim.o: sim.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

clean:; -rm \
	sim sim.o awall.o \
	awall_dev.o awall_hst.o \
	libawall_hst.a libawall_dev.a

.PHONY: clean all
