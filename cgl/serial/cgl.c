/*
Author: Jason He
Version: 1.0 20210521 Serial version.
*/

#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <time.h>
#include <math.h>

/*
Compute problem xxx in textbook on a 2D grid. Terminal time T=10000.

Inputs:
   N: the size of the grid in both directions
   c1: double precision parameter
   c3: double precision parameter
   M: number of time steps
   s: long integer seed value(optional)

Outputs:
	print arguments including the seed
	output grid value into "BrussU.out" and "BrussV.out" that contain
		the data at t=100k, k=0,1,2,...,10.
	print run time and output it to runtime.dat

E.g.
$ ./cgl 128 1.5 0.25 100000 12345

*/

void spectrald(fftw_complex *B, int N)//spectral derivative
{
   for(int j=0; j<N; j++)
   {
      int jj = (j <= N/2 ? j : j-N);//the mode index
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

void RK4(fftw_complex *A, fftw_complex *B, fftw_complex *C, int N, double dt, double C1r, double C1i, double c3, double tcoeff)
{
   for(int i=0; i<N; i++)
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
   
   //load input parameters
   int argi = 0;
   int N = atoi(argv[++argi]);
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

   printf( "N = %d\nc1 = %lf\nc3 = %lf\nM = %d\n"
      "Starting seed = %ld\n", N, c1, c3, M, seed);
   
   //parameters for calculation. refer to textbook Eq(37.15)
   double dt = 10000.0/M;//terminal time = 10000
   double C1r = 1.0/64/64;//L = 128pi
   double C1i = C1r*c1;
   
   //Three sets of grid points
   fftw_complex *A = fftw_malloc(N*N*sizeof(fftw_complex));//A is the complex field value
   fftw_complex *B = fftw_malloc(N*N*sizeof(fftw_complex));//for spectral decomposition
   fftw_complex *C = fftw_malloc(N*N*sizeof(fftw_complex));//for intermediate step in RK4 method
   for(int i=0; i<N*N; i++)
   {
      A[i][0] = 3*drand48() -1.5;//set initial value of A
      A[i][1] = 3*drand48() -1.5;
      C[i][0] = A[i][0];//set initial value of C = A
      C[i][1] = A[i][1];
   }
   
   //file output
   FILE* file = fopen("CGL.out","w");
   fwrite(A, sizeof(fftw_complex), N*N, file);
   printf("Saved output at t=0\n");
   
   //FFTW plans
   fftw_plan p = fftw_plan_dft_2d(N, N, C, B, FFTW_FORWARD, FFTW_ESTIMATE);
   fftw_plan pinv = fftw_plan_dft_2d(N, N, B, B, FFTW_BACKWARD, FFTW_ESTIMATE);
   
   float runtime;
   clock_t t;
   t = clock();
   //main loop
   for(int step=0; step<M; ++step)
   {
      //first step in RK4
      fftw_execute(p);
      spectrald(B,N);
      fftw_execute(pinv);//laplacian stored in B
      RK4(A, B, C, N, dt, C1r, C1i, c3, 0.25);
      
      //second step in RK4
      fftw_execute(p);
      spectrald(B,N);
      fftw_execute(pinv);//laplacian stored in B
      RK4(A, B, C, N, dt, C1r, C1i, c3, 1.0/3);
      
      //third step in RK4
      fftw_execute(p);
      spectrald(B,N);
      fftw_execute(pinv);//laplacian stored in B
      RK4(A, B, C, N, dt, C1r, C1i, c3, 0.5);
      
      //fourth step in RK4
      fftw_execute(p);
      spectrald(B,N);
      fftw_execute(pinv);//laplacian stored in B
      RK4(A, B, C, N, dt, C1r, C1i, c3, 1.0);
      
      //store the final value of this time step to A
      for(int i=0; i<N*N; i++)
      {
         A[i][0] = C[i][0];
         A[i][1] = C[i][1];
      }
      
      //output to files
      if((step+1)%(M/10) == 0)
      {
         fwrite(A, sizeof(fftw_complex), N*N, file);
         printf("Saved output at t=%d\n", (step+1)/10);
      }
   }//main loop
   //print runtime and output to the file
   runtime = (float)(clock()-t)/CLOCKS_PER_SEC;
   FILE* tfile = fopen("runtime.dat", "a");
   fprintf(tfile, "%d %f\n", N, runtime);
   printf("Runtime = %f seconds\n", runtime);
   
   //close files and free memories
   fclose(tfile);
   fclose(file);
   fftw_destroy_plan(p);
   fftw_destroy_plan(pinv);
   fftw_free(A);
   fftw_free(B);
   fftw_free(C);
   
   return 0;
}