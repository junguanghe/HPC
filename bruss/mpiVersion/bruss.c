/*
Author: Jason He
Version: 1.0 20210422
Version: 1.1 20210501 Fixed an error in dealing with boundary conditions.
                      the code is correct now.
Version: 1.2 20210518 Use the LAPACK library instead of LAPACKE since it's
                      not installed on Northwestern Quest cluster.
Version: 1.3 20210519 Redefine x to be the contiguous direction(row), for
                      the sake of gathering data from different processes
                      in future MPI version.
Version: 1.4 20210520 MPI version using MPI_Allgather.
Version: 1.5 20210716 Fix dx=N-1, i.e. N-1 intervals between N grid points
*/

#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
//#include <lapack.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
extern void dgttrf_();
extern void dgttrs_();

/*
This program solve the Brusselator Model- 35.4.1 in the textbook. Ternimal time
T=1000.

Inputs: 
	number of grid points in each direction, N.
	double precision diffusion coefficients, Du, Dv
	double precision fix point coefficients, A, B
	number of time steps, M
	(optional) long integer seed value generating random initial condition

Outputs:
	print arguments including the seed
	output grid value into "BrussU.out" and "BrussV.out" that contain
		the data at t=100k, k=0,1,2,...,10.
	print run time and output it to runtime.dat

E.g.
$ mpirun -np 1 bruss 128 5.0e-5 5.0e-6 1 3 10000 12345

*/

int main(int argc, char* argv[])
{
   // Initialize MPI
   MPI_Init(&argc, &argv);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int size;
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   int data;
   
   //load input parameters
   int argi = 0;
   int N = atoi(argv[++argi]);
   double Du = atof(argv[++argi]);
   double Dv = atof(argv[++argi]);
   double A = atof(argv[++argi]);
   double B = atof(argv[++argi]);
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
      printf( "N = %d\nDu = %lf\nDv = %lf\n"
         "A = %lf\nB = %lf\nM = %d\n"
         "Starting seed = %ld\n"
         "Number of cores = %d\n", N,
         Du, Dv, A, B, M, seed, size);
   }
	
   //Initialize the tridiagonal matrices
   double dt = 1000.0/M; // Terminal time = 1000
   double dx = 1.0/(N-1);    // grid is from 0 to 1 in x and y direction
   double Udiag = 1 + dt*Du/dx/dx;
   double Vdiag = 1 + dt*Dv/dx/dx;
   double Uoffdiag = -0.5*dt*Du/dx/dx;
   double Voffdiag = -0.5*dt*Dv/dx/dx;
   double* Uld = malloc(N*sizeof(double));
   double* Ud = malloc(N*sizeof(double));
   double* Uud = malloc(N*sizeof(double));
   double* Uuud = malloc(N*sizeof(double));
   double* Vld = malloc(N*sizeof(double));
   double* Vd = malloc(N*sizeof(double));
   double* Vud = malloc(N*sizeof(double));
   double* Vuud = malloc(N*sizeof(double));
   for(int i=0; i<N; ++i)
   {
      Uld[i] = Uoffdiag;
      Ud[i] = Udiag;
      Uud[i] = Uoffdiag;
      Vld[i] = Voffdiag;
      Vd[i] = Vdiag;
      Vud[i] = Voffdiag;
   }
   Uld[N-1] = 0;
   Uld[N-2] = 0;
   Ud[N-1] = 1;
   Ud[0] = 1;
   Uud[N-1] = 0;
   Uud[0] = 0;
   Vld[N-1] = 0;
   Vld[N-2] = 0;
   Vd[0] = 1;
   Vd[N-1] = 1;
   Vud[N-1] = 0;
   Vud[0] = 0;
   
   //parameters for dgttrf and dgttrs solver
   int n = N;
   int* Uipiv = malloc(N*sizeof(int));
   int Uinfo = 0;
   dgttrf_(&n, Uld, Ud, Uud, Uuud, Uipiv, &Uinfo);
   int* Vipiv = malloc(N*sizeof(int));
   int Vinfo = 0;
   dgttrf_(&n, Vld, Vd, Vud, Vuud, Vipiv, &Vinfo);
   char trT = 'T';
   char trN = 'N';
   int nrhs = N-2; // -2 due to boundary condition
   
   //two sets of grid points
   double (*u)[N] = malloc(N*sizeof(*u)); // u and v are for temporal updates
   double (*v)[N] = malloc(N*sizeof(*v));
   double (*U)[N] = malloc(N*sizeof(*U));
   double (*V)[N] = malloc(N*sizeof(*V));

   for(int i=0; i<N; i++)
   {
      for(int j=0; j<N; j++)
      {
         u[i][j] = A + 2*drand48() -1; // set initial value of u and v
         v[i][j] = B + 2*drand48() -1;
         U[i][j] = u[i][j]; // set initial value of U = u and V = v
         V[i][j] = v[i][j];
      }
   }
   
   //file output at rank0
   FILE* ufile = fopen("BrussU.out","w");
   FILE* vfile = fopen("BrussV.out","w");
   if(rank==0)
   {
      fwrite(U, sizeof(double), N*N, ufile);
      fwrite(V, sizeof(double), N*N, vfile);
      printf("Saved output at t=0\n");   
   }
   
   //domain decomposition
   int rmder = N%size; //number of extra rows if N is not
                       //evenly divisible by size
   int *counts = malloc(size*sizeof(int));//Track counts and offets
   int *offsets = malloc(size*sizeof(int));//for Allgatherv function
   for(int i=0; i<size; ++i)
   {
      counts[i] = N*(N/size + (rmder > i ? 1 : 0));
      offsets[i] = N*(N/size*i + (rmder > i ? i : rmder));
   }
   int localN = N/size + (rmder > rank ? 1 : 0);//number of rows
                                                //in the local process
   int locali = N/size*rank + (rmder > rank ? rank : rmder);
   int nexti = locali + localN;//starting row index of the local and next process

   float runtime;
   clock_t t;
   t = clock();
   //main loop
   for(int step=0; step<M; ++step)
   {
      // First step: Explicit in x direction
      for(int i=locali; i<nexti; ++i)
      {
         u[i][0] = U[i][0];
         v[i][0] = V[i][0];// boundary conditions
         for(int j=1; j<N-1; ++j)
         {
            u[i][j] = U[i][j] - Uoffdiag*(U[i][j+1] - 2*U[i][j] + U[i][j-1])
                      + 0.5*dt*(A + U[i][j]*U[i][j]*V[i][j] - (B+1)*U[i][j]);
            v[i][j] = V[i][j] - Voffdiag*(V[i][j+1] - 2*V[i][j] + V[i][j-1])
                      + 0.5*dt*(B*U[i][j] - U[i][j]*U[i][j]*V[i][j]); //Uoffdiag<0 so that term is '-'
         }
         u[i][N-1] = U[i][N-1];
         v[i][N-1] = V[i][N-1];// boundary conditions
      }
      for(int j=0; j<N; ++j)
      {
         u[0][j] = U[0][j];
         v[0][j] = V[0][j];
         u[N-1][j] = U[N-1][j];
         v[N-1][j] = V[N-1][j];// boundary conditions
      }
      
      //Allgather the data
      MPI_Allgatherv(&(u[locali][0]), localN, MPI_DOUBLE, &(u[0][0]), counts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
      MPI_Allgatherv(&(v[locali][0]), localN, MPI_DOUBLE, &(v[0][0]), counts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
      
      //Transpose the RHS
      for(int i=0; i<N; ++i)
      {
         for(int j=0; j<N; ++j)
         {
            U[i][j] = u[j][i];
            V[i][j] = v[j][i];
         }
      }

      //Second step: Implicit in y direction
      dgttrs_(&trN, &n, &localN, Uld, Ud, Uud, Uuud, Uipiv, &(U[locali][0]), &n, &Uinfo);
      dgttrs_(&trN, &n, &localN, Vld, Vd, Vud, Vuud, Vipiv, &(V[locali][0]), &n, &Vinfo); // start from V[locali][0] with nrhs=localN columns
      for(int j=0; j<N; ++j)
      {
         U[0][j] = u[j][0];
         V[0][j] = v[j][0];
         U[N-1][j] = u[j][N-1];
         V[N-1][j] = v[j][N-1];// boundary conditions
      }
      
      //Third step: Explicit in y direction
      for(int i=locali; i<nexti; ++i)
      {
         u[i][0] = U[i][0];
         v[i][0] = V[i][0];// boundary conditions
         for(int j=1; j<N-1; ++j)
         {
            u[i][j] = U[i][j] - Uoffdiag*(U[i][j+1] - 2*U[i][j] + U[i][j-1])
                      + 0.5*dt*(A + U[i][j]*U[i][j]*V[i][j] - (B+1)*U[i][j]);
            v[i][j] = V[i][j] - Voffdiag*(V[i][j+1] - 2*V[i][j] + V[i][j-1])
                      + 0.5*dt*(B*U[i][j] - U[i][j]*U[i][j]*V[i][j]); //Uoffdiag<0 so that term is '-'
         }
         u[i][N-1] = U[i][N-1];
         v[i][N-1] = V[i][N-1];// boundary conditions
      }
      for(int j=0; j<N; ++j)
      {
         u[0][j] = U[0][j];
         v[0][j] = V[0][j];
         u[N-1][j] = U[N-1][j];
         v[N-1][j] = V[N-1][j];// boundary conditions
      }
      
      //Allgather the data
      MPI_Allgatherv(&(u[locali][0]), localN, MPI_DOUBLE, &(u[0][0]), counts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
      MPI_Allgatherv(&(v[locali][0]), localN, MPI_DOUBLE, &(v[0][0]), counts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
      
      //Transpose back the RHS
      for(int i=0; i<N; ++i)
      {
         for(int j=0; j<N; ++j)
         {
            U[i][j] = u[j][i];
            V[i][j] = v[j][i];
         }
      }

      //Fourth step: Implicit in x direction
      dgttrs_(&trN, &n, &localN, Uld, Ud, Uud, Uuud, Uipiv, &(U[locali][0]), &n, &Uinfo);
      dgttrs_(&trN, &n, &localN, Vld, Vd, Vud, Vuud, Vipiv, &(V[locali][0]), &n, &Vinfo); // start from V[locali][0] with nrhs=localN columns
      for(int j=0; j<N; ++j)
      {
         U[0][j] = u[j][0];
         V[0][j] = v[j][0];
         U[N-1][j] = u[j][N-1];
         V[N-1][j] = v[j][N-1];// boundary conditions
      }
      
      //output to files at rank0
      if((step+1)%(M/10) == 0)
      {
         //Allgather the data
         MPI_Allgatherv(&(U[locali][0]), localN, MPI_DOUBLE, &(U[0][0]), counts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
         MPI_Allgatherv(&(V[locali][0]), localN, MPI_DOUBLE, &(V[0][0]), counts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
         if(rank==0)
         {
            fwrite(U, sizeof(double), N*N, ufile);
            fwrite(V, sizeof(double), N*N, vfile);
            printf("Saved output at t=%d\n", (step+1)/10);
         }
      }
   }//main loop
   
   //print runtime output to the file at rank0
   if(rank==0)
   {
      runtime = (float)(clock()-t)/CLOCKS_PER_SEC;
      FILE* tfile = fopen("runtime.dat", "a");
      fprintf(tfile, "%d %d %f\n", N, size, runtime);
      printf("Runtime = %f seconds\n", runtime);
      fclose(tfile);
   }
   
   //close files and free memories
   fclose(ufile);
   fclose(vfile);
   free(u);
   free(v);
   free(U);
   free(V);
   free(Uld);
   free(Ud);
   free(Uud);
   free(Uuud);
   free(Vld);
   free(Vd);
   free(Vud);
   free(Vuud);
   
   //Finalize MPI
   MPI_Finalize();
   return 0.;
}
