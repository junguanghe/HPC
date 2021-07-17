ID = -I/software/FFTW/3.3.3-RH7/include
LD = -L/software/FFTW/3.3.3-RH7/lib
cgl: cgl.c
	mpicc -std=c99 -c $(ID) cgl.c
	mpicc -std=c99 -o cgl $(LD) cgl.o -lfftw3_mpi -lfftw3 -lm
clean:
	-rm cgl cgl.o