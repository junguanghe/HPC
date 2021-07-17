/*
Author: Jason He
Version: 1.0 20210521 Serial version.
Version: 2.0 20210523 MPI version using parallel fftw.
*/
#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <fftw3-mpi.h>

/*
Compute problem xxx in textbook on a 2D grid. Terminal time T=10000.

Inputs:
   N: the size of the grid in both directions(should be evenly divisible by the number of processes)
   c1: double precision parameter
   c3: double precision parameter
   M: number of time steps
   s: long integer seed value(optional)

Outputs:
	print arguments including the seed
	output grid value into "BrussU.out" and "BrussV.out" that contain
		the data at t=100k, k=0,1,2,...,10.
	print run time and output it to runtime.dat

To compile on quest,
$ module load mpi/openmpi-1.10.5-gcc-4.8.3
$ module load fftw/3.3.3-gcc

To run, E.g.
$ mpirun -np 1 cgl 128 1.5 0.25 100000 12345

*/

void spectrald(fftw_complex *B, int N, ptrdiff_t localn, ptrdiff_t local0)//spectral derivative
{
   for(int j=0; j<localn; j++)
   {
      int jj = (j+local0 <= N/2 ? j+local0 : j+local0-N);//the mode index
      for(int k=0; k<N; k++)
      {
         int kk = (k <= N/2 ? k : k-N);//the mode index
         int i = j*N + k;
         B[i][0] *= -(jj*jj + kk*kk);//do the spectral derivative
         B[i][1] *= -(jj*jj + kk*kk);
      }
   }
   return;
}

void RK4(fftw_complex *A, fftw_complex *B, fftw_complex *C, int N, double dt, double C1r, double C1i, double c3, ptrdiff_t localn, double tcoeff)
{
   for(int i=0; i<localn; i++)
   {
      for(int j=0; j<N; j++)
      {
         int k = i*N+j;
         double C2 = C[k][0]*C[k][0] + C[k][1]*C[k][1];
         double tmp = C[k][0];//calculate the real part fisrt, so store the original value temporarily
         C[k][0] = A[k][0] + tcoeff*dt*(C[k][0] + (C1r*B[k][0] - C1i*B[k][1])/N/N - C2*(C[k][0] + c3*C[k][1]));
         C[k][1] = A[k][1] + tcoeff*dt*(C[k][1] + (C1r*B[k][1] + C1i*B[k][0])/N/N - C2*(C[k][1] - c3*tmp));//devide the extra N^2 by forward FFT
      }
   }//intermediate step result stored in C
}

int main(int argc, char* argv[])
{
#ifndef M_PI
   const double M_PI = 4.0*atan(1.0);
#endif
   
   //initialize MPI
   MPI_Init(&argc, &argv);
   fftw_mpi_init();
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int size;
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   
   //for measurement of the compute time
   double precision = MPI_Wtick();
   double starttime = MPI_Wtime();
   
   //load input parameters
   int argi = 0;
   const ptrdiff_t N = atol(argv[++argi]);
   double c1 = atof(argv[++argi]);
   double c3 = atof(argv[++argi]);
   int M = atoi(argv[++argi]);
   long int seed;
   if (argi < argc-1)
   {
      seed = atol(argv[++argi]);
   }
   else
   {
      seed = (long int)time(NULL);
   }
   srand48(seed);

   if(rank==0)
   {
      printf( "N = %ld\nc1 = %lf\nc3 = %lf\nM = %d\n"
         "Starting seed = %ld\nnumber of processes = %d\n", N, c1, c3, M, seed, size);
   }
   
   //parameters for calculation. refer to textbook Eq(37.15)
   double dt = 10000.0/M;//terminal time = 10000
   double C1r = 1.0/64/64;//L = 128pi
   double C1i = C1r*c1;
   
   //parallel fft parameters
   ptrdiff_t localn, local0;
   ptrdiff_t alloc_local = fftw_mpi_local_size_2d(N, N, MPI_COMM_WORLD, &localn, &local0);
   //set up three sets of grid points
   fftw_complex *A = fftw_alloc_complex(alloc_local);//A is the complex field value
   fftw_complex *B = fftw_alloc_complex(alloc_local);//for spectral decomposition
   fftw_complex *C = fftw_alloc_complex(alloc_local);//for intermediate step in RK4 method
   //sequentially initialize value of the grid for each processes
   for(int r=0; r<size; r++)
   {
      if(rank==r)
      {
         for(int i=0; i<N*localn; i++)
         {
            A[i][0] = 3*drand48() -1.5;//set initial value of A
            A[i][1] = 3*drand48() -1.5;
            C[i][0] = A[i][0];//set initial value of C = A
            C[i][1] = A[i][1];
         }
      }
      else
      {
         for(int i=0; i<N*localn; i++)
         {
            drand48();//discard random values for other processes
            drand48();
         }
      }
   }
   
   //set up the whole grid in rank0 and gatter initial value from each processes
   double *a = NULL;
   if(rank==0)
   {
      a = (double*)malloc(N*N*sizeof(fftw_complex));
   }
   MPI_Gather(A, 2*localn*N, MPI_DOUBLE, a, 2*localn*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   
   //file output at rank0
   FILE* file = fopen("CGL.out","w");
   if(rank==0)
   {
      fwrite(a, sizeof(fftw_complex), N*N, file);
      printf("Saved output at t=0\n");
   }
   
   //FFTW plans
   fftw_plan p = fftw_mpi_plan_dft_2d(N, N, C, B, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
   fftw_plan pinv = fftw_mpi_plan_dft_2d(N, N, B, B, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
   
   //main loop
   for(int step=0; step<M; ++step)
   {
      //first step in RK4
      fftw_execute(p);
      spectrald(B,N,localn,local0);
      fftw_execute(pinv);//laplacian stored in B
      RK4(A, B, C, N, dt, C1r, C1i, c3, localn, 0.25);
      
      //second step in RK4
      fftw_execute(p);
      spectrald(B,N,localn,local0);
      fftw_execute(pinv);//laplacian stored in B
      RK4(A, B, C, N, dt, C1r, C1i, c3, localn, 1.0/3);
      
      //third step in RK4
      fftw_execute(p);
      spectrald(B,N,localn,local0);
      fftw_execute(pinv);//laplacian stored in B
      RK4(A, B, C, N, dt, C1r, C1i, c3, localn, 0.5);
      
      //fourth step in RK4
      fftw_execute(p);
      spectrald(B,N,localn,local0);
      fftw_execute(pinv);//laplacian stored in B
      RK4(A, B, C, N, dt, C1r, C1i, c3, localn, 1.0);
      
      //store the final value of this time step to A
      for(int i=0; i<N*localn; i++)
      {
         A[i][0] = C[i][0];
         A[i][1] = C[i][1];
      }
      
      //output to files
      if((step+1)%(M/10) == 0)
      {
         MPI_Gather(A, 2*localn*N, MPI_DOUBLE, a, 2*localn*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
         if(rank==0)
         {
            fwrite(a, sizeof(fftw_complex), N*N, file);
            printf("Saved output at t=%d\n", (step+1)/10);
         }
      }
   }//main loop
   
   //close files and free memories
   fclose(file);
   fftw_destroy_plan(p);
   fftw_destroy_plan(pinv);
   fftw_free(A);
   fftw_free(B);
   fftw_free(C);
   if(rank==0) fftw_free(a);
   
   //print runtime and output to the file
   double time_elapsed = MPI_Wtime() - starttime;
   if(rank==0)
   {
      FILE* tfile = fopen("runtime.dat", "a");
      fprintf(tfile, "%d %d %lf\n", N, size, time_elapsed);
      printf("Execution time = %lf seconds, with precision %le seconds.\n", time_elapsed, precision);
      fclose(tfile);
   }
   
   MPI_Finalize();
   
   return 0;
}