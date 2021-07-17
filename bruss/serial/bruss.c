/*
Author: Jason He
Version: 1.0 20210422
Version: 1.1 20210501 Fixed an error in dealing with boundary conditions.
                      the code is correct now.
Version: 2.0 20210502 Make use of the LAPACK_ROW_MAJOR option, and get rid of
                      the manual transpose of the RHS in the implicit x-direction step.
                      Add more detailed comments.
*/

#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <time.h>
#include <string.h>

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
   lapack_int n = N;
   lapack_int* Uipiv = malloc(N*sizeof(lapack_int));
   lapack_int Uinfo = 0;
   Uinfo = LAPACKE_dgttrf(n, Uld, Ud, Uud, Uuud, Uipiv);
   lapack_int* Vipiv = malloc(N*sizeof(lapack_int));
   lapack_int Vinfo = 0;
   Vinfo = LAPACKE_dgttrf(n, Vld, Vd, Vud, Vuud, Vipiv);
   char trT = 'T';
   char trN = 'N';
   lapack_int nrhs = N-2; // -2 due to boundary condition
   
   //two sets of grid points
   double (*u)[N] = malloc(N*sizeof(*u));
   double (*v)[N] = malloc(N*sizeof(*v));
   double (*U)[N] = malloc(N*sizeof(*U)); // U and V are for temporal updates
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
      for(int j=1; j<N-1; ++j)
      {
         for(int i=1; i<N-1; ++i)
         {
            U[i][j] = u[i][j] - Uoffdiag*(u[i+1][j] - 2*u[i][j] + u[i-1][j])
                      + 0.5*dt*(A + u[i][j]*u[i][j]*v[i][j] - (B+1)*u[i][j]);
            V[i][j] = v[i][j] - Voffdiag*(v[i+1][j] - 2*v[i][j] + v[i-1][j])
                      + 0.5*dt*(B*u[i][j] - u[i][j]*u[i][j]*v[i][j]); //Uoffdiag<0 so that term is '-'
         }
      }
      
      //Second step: Implicit in y direction
      Uinfo = LAPACKE_dgttrs(LAPACK_COL_MAJOR, trN, n, nrhs, Uld, Ud, Uud, Uuud,
                              Uipiv, &(U[1][0]), n);
      Vinfo = LAPACKE_dgttrs(LAPACK_COL_MAJOR, trN, n, nrhs, Vld, Vd, Vud, Vuud,
                              Vipiv, &(V[1][0]), n); // start from v[1][0] with nrhs=N-2 columns
      // Although the implicit step has Dirichlet-type condition, to be safe, we still manually
      // fix the boundaries.
      for(int i=1; i<N-1; ++i)
      {
         U[i][0] = u[i][0];
         V[i][0] = v[i][0];
         U[i][N-1] = u[i][N-1];
         V[i][N-1] = v[i][N-1];// boundary conditions
      }
      
      //Third step: Explicit in y direction
      for(int i=1; i<N-1; ++i)
      {
         for(int j=1; j<N-1; ++j)
         {
            u[i][j] = U[i][j] - Uoffdiag*(U[i][j+1] - 2*U[i][j] + U[i][j-1])
                      + 0.5*dt*(A + U[i][j]*U[i][j]*V[i][j] - (B+1)*U[i][j]);
            v[i][j] = V[i][j] - Voffdiag*(V[i][j+1] - 2*V[i][j] + V[i][j-1])
                      + 0.5*dt*(B*U[i][j] - U[i][j]*U[i][j]*V[i][j]); //Uoffdiag<0 so that term is '-'
         }
      }
      
      //Fourth step: Implicit in x direction
      Uinfo = LAPACKE_dgttrs(LAPACK_ROW_MAJOR, trN, n, nrhs+2, Uld, Ud, Uud, Uuud,
                              Uipiv, &(u[0][0]), n);
      Vinfo = LAPACKE_dgttrs(LAPACK_ROW_MAJOR, trN, n, nrhs+2, Vld, Vd, Vud, Vuud,
                              Vipiv, &(v[0][0]), n); // start from v[0][0] with nrhs+2=N columns
      // If we make use of the LAPACK_ROW_MAJOR, we will have to put in the whole matrices u and v,
      // and we have to keep the boundaries unchanged.
      for(int k=0; k<N; ++k)
      {
         u[k][0] = U[k][0];
         v[k][0] = V[k][0];
         u[k][N-1] = U[k][N-1];
         v[k][N-1] = V[k][N-1];
         u[0][k] = U[0][k];
         v[0][k] = V[0][k];
         u[N-1][k] = U[N-1][k];
         v[N-1][k] = V[N-1][k];// boundary conditions
      }
      
      //output to files
      if((step+1)%(M/10) == 0)
      {
         fwrite(u, sizeof(double), N*N, ufile);
         fwrite(v, sizeof(double), N*N, vfile);
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
