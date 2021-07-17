/*
Author: Jason He
Version: 1.0 20210605 Serial version.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*
This program solve the Stokes Flow problem- 36.1.2 in the textbook.

Inputs: 
	N, number of grid points in each direction.
	mu, double precision viscosity.
	P, double precision pressure drop.
	w, double precision relaxation parameter.
    tol, double precision error tolerance.
	K, maximum number of iterations.

Outputs:
	print arguments
	output final grid value into "StokesU.out", "StokesV.out" and "StokesP.out"
	print run time and output it to runtime.dat

E.g.
$ ./stokes 128 1 1 0.4 1e-9 100000

*/

//declare global variables
int N;
double mu;
double P;
double w;
double tol;
int K;
double dx;

//return the bigger one of two doubles
double max(double a, double b){
    return (a > b ? a : b);
}

//the three update functions update the grid values and return the maximum residual
double updateU(double (*u)[N-1], double (*p)[N-1]){
    double maxres=0.;
    double r;

    r = mu*(u[1][0]+u[0][1]-4*u[0][0]) - dx*2*(p[0][0]-P);
    maxres = max(maxres, fabs(r));
    u[0][0] += w*r;//update u at (0,0)

    for(int k=1; k<N-2; k++){
        r = mu*(u[1][k]+u[0][k+1]+u[0][k-1]-3*u[0][k]) - dx*2*(p[0][k]-P);
        maxres = max(maxres, fabs(r));
        u[0][k] += w*r;//update u from (0,1) to (0,N-3)
    }
    
    r = mu*(u[1][N-2]+u[0][N-3]-4*u[0][N-2]) - dx*2*(p[0][N-2]-P);
    maxres = max(maxres, fabs(r));
    u[0][N-2] += w*r;//update u at (0,N-2)

    for(int j=1; j<N-1; j++){
        r = mu*(u[j+1][0]+u[j-1][0]+u[j][1]-5*u[j][0]) - dx*(p[j][0]-p[j-1][0]);
        maxres = max(maxres, fabs(r));
        u[j][0] += w*r;//update u from (1,0) to (N-2,0)

        for(int k=1; k<N-2; k++){
            r = mu*(u[j+1][k]+u[j-1][k]+u[j][k+1]+u[j][k-1]-4*u[j][k])
                - dx*(p[j][k]-p[j-1][k]);
            maxres = max(maxres, fabs(r));
            u[j][k] += w*r;// update u in bulk
        }

        r = mu*(u[j+1][N-2]+u[j-1][N-2]+u[j][N-3]-5*u[j][N-2]) - dx*(p[j][N-2]-p[j-1][N-2]);
        maxres = max(maxres, fabs(r));
        u[j][N-2] += w*r;//update u from (1,N-2) to (N-2,N-2)
    }

    r = mu*(u[N-2][0]+u[N-1][1]-4*u[N-1][0]) - dx*2*(0-p[N-2][0]);
    maxres = max(maxres, fabs(r));
    u[N-1][0] += w*r;//update u at (N-1,0)

    for(int k=1; k<N-2; k++){
        r = mu*(u[N-2][k]+u[N-1][k+1]+u[N-1][k-1]-3*u[N-1][k]) - dx*2*(0-p[N-2][k]);
        maxres = max(maxres, fabs(r));
        u[N-1][k] += w*r;//update u from (N-1,1) to (N-1,N-3)
    }
    
    r = mu*(u[N-2][N-2]+u[N-1][N-3]-4*u[N-1][N-2]) - dx*2*(0-p[N-2][N-2]);
    maxres = max(maxres, fabs(r));
    u[N-1][N-2] += w*r;//update u at (N-1,N-2)

    return maxres;
}

double updateV(double (*v)[N], double (*p)[N-1]){
    double maxres=0.;
    double r;

    for(int k=1; k<N-1; k++){
        r = mu*(v[0][k+1]+v[0][k-1]+v[1][k]-3*v[0][k]) - dx*(p[0][k]-p[0][k-1]);
        maxres = max(maxres, fabs(r));
        v[0][k] += w*r;//update v from (0,1) to (0,N-2)

        for(int j=1; j<N-2; j++){
            r = mu*(v[j+1][k]+v[j-1][k]+v[j][k+1]+v[j][k-1]-4*v[j][k])
                - dx*(p[j][k]-p[j][k-1]);
            maxres = max(maxres, fabs(r));
            v[j][k] += w*r;//update v in bulk
        }

        r = mu*(v[N-2][k+1]+v[N-2][k-1]+v[N-3][k]-3*v[N-2][k]) - dx*(p[N-2][k]-p[N-2][k-1]);
        maxres = max(maxres, fabs(r));
        v[N-2][k] += w*r;//update v from (N-2,1) to (N-2,N-2)

    }

    return maxres;
}

double updateP(double (*u)[N-1], double (*v)[N], double (*p)[N-1]){
    double maxres=0.;
    double r;
    for(int j=0; j<N-1; j++){
        for(int k=0; k<N-1; k++){
            r = -(u[j+1][k]-u[j][k])-(v[j][k+1]-v[j][k]);
            maxres = max(maxres, fabs(r));
            p[j][k] += w*r;//update all p's
        }
    }
    return maxres;
}

int main(int argc, char* argv[]){
    //load input parameters
    int argi = 0;
    N = atoi(argv[++argi]);
    mu = atof(argv[++argi]);
    P = atof(argv[++argi]);
    w = atof(argv[++argi]);
    tol = atof(argv[++argi]);
    K = atoi(argv[++argi]);
    printf( "N = %d\nmu = %lf\nP = %lf\nw = %lf\ntau = %e\nK = %d\n",
            N, mu, P, w, tol, K);
    
    //calculate parameters for the iteration equation
    dx = 1./(N-1);
    
    //initialize the grids
    double (*u)[N-1] = malloc(N*sizeof(*u));
    double (*v)[N] = malloc((N-1)*sizeof(*v));
    double (*p)[N-1] = malloc((N-1)*sizeof(*p));
    memset(&(u[0][0]), 0, sizeof(u));
    memset(&(v[0][0]), 0, sizeof(v));
    memset(&(p[0][0]), 0, sizeof(p));

    int iter=0;//number of iteration
    double maxres=1.0;//maximum residual of u,v,p on every grid points
    double ures;
    double vres;
    double pres;
    float runtime;//record runtime
    clock_t t;
    t = clock();
    //main loop
    while(iter<K && maxres>tol){
        ures = updateU(u, p);
        vres = updateV(v, p);
        pres = updateP(u, v, p);
        maxres = max(max(ures, vres), pres);
        iter++;
    }
    runtime = (float)(clock()-t)/CLOCKS_PER_SEC;
    FILE* tfile = fopen("runtime.dat", "a");
    fprintf(tfile, "%d %f\n", N, runtime/iter);
    fclose(tfile);
    printf("Runtime = %f seconds\nOne iteration runtime = %f seconds\n",
            runtime, runtime/iter);

    //print iterations and residuals
    printf("number of iterations = %d\nresidual = %e\n", iter, maxres);

    //output to files
    FILE* ufile = fopen("stokesU.out","w");
    fwrite(u, sizeof(double), N*(N-1), ufile);
    fclose(ufile);
    FILE* vfile = fopen("stokesV.out","w");
    fwrite(v, sizeof(double), (N-1)*N, vfile);
    fclose(vfile);
    FILE* pfile = fopen("stokesP.out","w");
    fwrite(p, sizeof(double), (N-1)*(N-1), pfile);
    fclose(pfile);

    //free memories
    free(u);
    free(v);
    free(p);

    return 0;
}