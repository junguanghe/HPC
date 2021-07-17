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
*/

#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
//#include <lapack.h>
#include <time.h>
#include <string.h>
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
$ ./bruss 128 5.0e-5 5.0e-6 1 3 10000 12345

*/

int main(int argc, char* argv[])
{
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

   printf( "N = %d\nDu = %lf\nDv = %lf\n"
      "A = %lf\nB = %lf\nM = %d\n"
      "Starting seed = %ld\n", N,
      Du, Dv, A, B, M, seed);
	
   //Initialize the tridiagonal matrices
   double dt = 1000.0/M; // Terminal time = 1000
   double dx = 1.0/N;    // grid is from 0 to 1 in x and y direction
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
   //Uinfo = LAPACKE_dgttrf(n, Uld, Ud, Uud, Uuud, Uipiv);
   dgttrf_(&n, Uld, Ud, Uud, Uuud, Uipiv, &Uinfo);
   int* Vipiv = malloc(N*sizeof(int));
   int Vinfo = 0;
   //Vinfo = LAPACKE_dgttrf(n, Vld, Vd, Vud, Vuud, Vipiv);
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
   
   //file output
   FILE* ufile = fopen("BrussU.out","w");
   FILE* vfile = fopen("BrussV.out","w");
   fwrite(u, sizeof(double), N*N, ufile);
   fwrite(v, sizeof(double), N*N, vfile);
   printf("Saved output at t=0\n");

   float runtime;
   clock_t t;
   t = clock();
   //main loop
   for(int step=0; step<M; ++step)
   {
      // First step: Explicit in x direction
      for(int j=0; j<N; ++j)
      {
         u[0][j] = U[0][j];
         v[0][j] = V[0][j];
         u[N-1][j] = U[N-1][j];
         v[N-1][j] = V[N-1][j];// boundary conditions
      }
      for(int i=1; i<N-1; ++i)
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
      dgttrs_(&trN, &n, &nrhs, Uld, Ud, Uud, Uuud, Uipiv, &(U[1][0]), &n, &Uinfo);
      dgttrs_(&trN, &n, &nrhs, Vld, Vd, Vud, Vuud, Vipiv, &(V[1][0]), &n, &Vinfo); // start from v[1][0] with nrhs=N-2 columns
      
      //Third step: Explicit in y direction
      for(int j=0; j<N; ++j)
      {
         u[0][j] = U[0][j];
         v[0][j] = V[0][j];
         u[N-1][j] = U[N-1][j];
         v[N-1][j] = V[N-1][j];// boundary conditions
      }
      for(int i=1; i<N-1; ++i)
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
      dgttrs_(&trN, &n, &nrhs, Uld, Ud, Uud, Uuud, Uipiv, &(U[1][0]), &n, &Uinfo);
      dgttrs_(&trN, &n, &nrhs, Vld, Vd, Vud, Vuud, Vipiv, &(V[1][0]), &n, &Vinfo); // start from v[1][0] with nrhs=N-2 columns
      
      //output to files
      if((step+1)%(M/10) == 0)
      {
         fwrite(U, sizeof(double), N*N, ufile);
         fwrite(V, sizeof(double), N*N, vfile);
         printf("Saved output at t=%d\n", (step+1)/10);
      }
   }//main loop
   //print runtime output to the file
   runtime = (float)(clock()-t)/CLOCKS_PER_SEC;
   FILE* tfile = fopen("runtime.dat", "a");
   fprintf(tfile, "%d %f\n", N, runtime);
   printf("Runtime = %f seconds\n", runtime);
   
   //close files and free memories
   fclose(tfile);
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
   
   return 0.;
}
