all: libhd_hst.a libhd_hst2.a

libhd_hst.a: hd.o; ar rv $@ $<
hd.o: hd.cpp

libhd_hst2.a: hd2.o; ar rv $@ $<
hd2.o: hd2.cpp

clean:; -rm hd.o libhd_hst.a hd2.o libhd_hst2.a 
