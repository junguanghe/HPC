/*
Author: Jason He
Version: 1.0 20210605 Serial version.
Version: 2.0 20210607 CUDA version.
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
const int threadsPerBlock = 1024;//for NVIDIA TESLA K20c maximum is 1024
/*the findmaxres kernel demand that threadsPerBlock should be a power of 2.
Also in the three update kernels, each block compare and store the residuals into
the global array dev_res[threadsPerBlock], which only work with maximum
threadsPerBlock=1024, since in this case only one block runs at a time.*/
__device__ __constant__ int dev_N;
__device__ __constant__ double dev_mu;
__device__ __constant__ double dev_P;
__device__ __constant__ double dev_w;
__device__ __constant__ double dev_dx;

//this kernel set values of 1d length-n double array to 0
__global__ void init(double* a, int n){
    int k = threadIdx.x + blockIdx.x*blockDim.x;
    if(k < n){
        a[k] = 0.;
    }
}

//the three update functions update the grid values and return the maximum residual
__global__ void updateU(double* dev_u, double* dev_p, double* dev_res, int color){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < dev_N*(dev_N-1)){
        int j = i/(dev_N-1);
        int k = i%(dev_N-1);
        if((j+k)%2==color){
            double r;
            int jp = i + dev_N-1;//(j+1)*(dev_N-1) + k;
            int jm = i - dev_N+1;//(j-1)*(dev_N-1) + k;
            int kp = i + 1;//j*(dev_N-1) + k+1;
            int km = i - 1;//j*(dev_N-1) + k-1;
            int pi = j*(dev_N-1) + k;
            int pm = pi - dev_N+1;//(j-1)*(dev_N-1) + k;
            if(j==0){
                if(k==0){
                    r = dev_mu*(dev_u[jp]+dev_u[kp]-4*dev_u[i])
                        -dev_dx*2*(dev_p[pi]-dev_P);
                }
                else if(k<dev_N-2){
                    r = dev_mu*(dev_u[jp]+dev_u[kp]+dev_u[km]-3*dev_u[i])
                        -dev_dx*2*(dev_p[pi]-dev_P);
                }
                else{
                    r = dev_mu*(dev_u[jp]+dev_u[km]-4*dev_u[i])
                        -dev_dx*2*(dev_p[pi]-dev_P);
                }
            }
            else if(j<dev_N-1){
                if(k==0){
                    r = dev_mu*(dev_u[jp]+dev_u[jm]+dev_u[kp]-5*dev_u[i])
                        -dev_dx*(dev_p[pi]-dev_p[pm]);
                }
                else if(k<dev_N-2){
                    r = dev_mu*(dev_u[jp]+dev_u[jm]+dev_u[kp]+dev_u[km]-4*dev_u[i])
                        -dev_dx*(dev_p[pi]-dev_p[pm]);
                }
                else{
                    r = dev_mu*(dev_u[jp]+dev_u[jm]+dev_u[km]-5*dev_u[i])
                        -dev_dx*(dev_p[pi]-dev_p[pm]);
                }
            }
            else{
                if(k==0){
                    r = dev_mu*(dev_u[jm]+dev_u[kp]-4*dev_u[i])
                        -dev_dx*2*(0-dev_p[pm]);
                }
                else if(k<dev_N-2){
                    r = dev_mu*(dev_u[jm]+dev_u[kp]+dev_u[km]-3*dev_u[i])
                        -dev_dx*2*(0-dev_p[pm]);
                }
                else{
                    r = dev_mu*(dev_u[jm]+dev_u[km]-4*dev_u[i])
                        -dev_dx*2*(0-dev_p[pm]);
                }
            }
            dev_u[i] += dev_w*r;
            dev_res[threadIdx.x] = max(fabs(r), dev_res[threadIdx.x]);
        }
    }
}

__global__ void updateV(double* dev_v, double* dev_p, double* dev_res, int color){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < (dev_N-1)*dev_N){
        int k = i%dev_N;
        if(0<k && k<dev_N-1){
            int j = i/dev_N;
            if((j+k)%2==color){
                double r;
                int jp = i + dev_N;//(j+1)*dev_N + k;
                int jm = i - dev_N;//(j-1)*dev_N + k;
                int kp = i + 1;//j*dev_N + k+1;
                int km = i - 1;//j*dev_N + k-1;
                int pi = j*(dev_N-1) + k;
                int pm = pi - 1;//j*(dev_N-1) + k-1;
                if(j==0){
                    r = dev_mu*(dev_v[kp]+dev_v[km]+dev_v[jp]-3*dev_v[i])
                        -dev_dx*(dev_p[pi]-dev_p[pm]);
                }
                else if(j<dev_N-2){
                    r = dev_mu*(dev_v[kp]+dev_v[km]+dev_v[jp]+dev_v[jm]-4*dev_v[i])
                        -dev_dx*(dev_p[pi]-dev_p[pm]);
                }
                else{
                    r = dev_mu*(dev_v[kp]+dev_v[km]+dev_v[jm]-3*dev_v[i])
                        -dev_dx*(dev_p[pi]-dev_p[pm]);
                }
                dev_v[i] += dev_w*r;
                dev_res[threadIdx.x] = max(fabs(r), dev_res[threadIdx.x]);
            }
        }
    }
}

__global__ void updateP(double* dev_u, double* dev_v, double* dev_p, double* dev_res){
    
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < (dev_N-1)*(dev_N-1)){
        int j = i/(dev_N-1);
        int k = i%(dev_N-1);
        int up = (j+1)*(dev_N-1) + k;
        int um = j*(dev_N-1) + k;
        int vp = j*dev_N + k+1;
        int vm = j*dev_N + k;
        double r;
        r = -(dev_u[up]-dev_u[um])-(dev_v[vp]-dev_v[vm]);
        dev_p[i] += dev_w*r;//update all p's
        dev_res[threadIdx.x] = max(fabs(r), dev_res[threadIdx.x]);//not comfortable with this
    }
}

//this kernel find the max value
__global__ void findmaxres(double* dev_res, int i){
    int k = threadIdx.x + blockIdx.x*blockDim.x;
    if(k%(2*i)==0){
        dev_res[k] = max(dev_res[k], dev_res[k+i]);
    }
}

int main(int argc, char* argv[]){
    //Choose gpu device
    cudaDeviceProp prop;
    int dev;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.multiProcessorCount = 13;
    cudaChooseDevice(&dev, &prop);
    cudaSetDevice(dev);

    //load input parameters
    int argi = 0;
    int N = atoi(argv[++argi]);
    double mu = atof(argv[++argi]);
    double P = atof(argv[++argi]);
    double w = atof(argv[++argi]);
    double tol = atof(argv[++argi]);
    int K = atoi(argv[++argi]);
    printf( "N = %d\nmu = %lf\nP = %lf\nw = %lf\ntau = %e\nK = %d\n",
            N, mu, P, w, tol, K);
    
    //calculate parameters for the iteration equation
    double dx = 1./(N-1);
    //calculate parameters for blocks and threads
    const int blocksPerGrid = (N*(N-1) + threadsPerBlock - 1)/threadsPerBlock;
    printf("BlocksPerGrid = %d\nThreadsPerBlock = %d\n", blocksPerGrid, threadsPerBlock);

    //copy parameters to device
    cudaMemcpyToSymbol(dev_N, &N, sizeof(double));
    cudaMemcpyToSymbol(dev_mu, &mu, sizeof(double));
    cudaMemcpyToSymbol(dev_P, &P, sizeof(double));
    cudaMemcpyToSymbol(dev_w, &w, sizeof(double));
    cudaMemcpyToSymbol(dev_dx, &dx, sizeof(double));

    //initialize value of the grid on device
    double *dev_u, *dev_v, *dev_p;
    cudaMalloc((void**)&dev_u, N*(N-1)*sizeof(double));
    cudaMalloc((void**)&dev_v, (N-1)*N*sizeof(double));
    cudaMalloc((void**)&dev_p, (N-1)*(N-1)*sizeof(double));
    init<<<blocksPerGrid, threadsPerBlock>>>(dev_u, N*(N-1));
    init<<<blocksPerGrid, threadsPerBlock>>>(dev_v, (N-1)*N);
    init<<<blocksPerGrid, threadsPerBlock>>>(dev_p, (N-1)*(N-1));
    
    //initialize grids of residual
    double *dev_res;
    cudaMalloc((void**)&dev_res, threadsPerBlock*sizeof(double));

    //initialize parameters for iterations
    int iter=0;//number of iteration
    double maxres=1.0;//maximum residual of u,v,p on every grid points
    float runtime;//record runtime
    clock_t t;
    t = clock();
    //main loop
    while(iter<K && maxres>tol){
        init<<<1, threadsPerBlock>>>(dev_res, threadsPerBlock);
        updateU<<<blocksPerGrid, threadsPerBlock>>>(dev_u, dev_p, dev_res, 0);
        updateU<<<blocksPerGrid, threadsPerBlock>>>(dev_u, dev_p, dev_res, 1);
        updateV<<<blocksPerGrid, threadsPerBlock>>>(dev_v, dev_p, dev_res, 0);
        updateV<<<blocksPerGrid, threadsPerBlock>>>(dev_v, dev_p, dev_res, 1);
        updateP<<<blocksPerGrid, threadsPerBlock>>>(dev_u, dev_v, dev_p, dev_res);
        for(int i=1; i<threadsPerBlock; i*=2)
            findmaxres<<<1, threadsPerBlock>>>(dev_res, i);
        cudaMemcpy(&maxres, dev_res, sizeof(double), cudaMemcpyDeviceToHost);
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
    
    //initialize the grids on the host
    double *u = (double*)malloc(N*(N-1)*sizeof(double));
    double *v = (double*)malloc((N-1)*N*sizeof(double));
    double *p = (double*)malloc((N-1)*(N-1)*sizeof(double));

    //output to files
    cudaMemcpy(u, dev_u, N*(N-1)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, dev_v, (N-1)*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(p, dev_p, (N-1)*(N-1)*sizeof(double), cudaMemcpyDeviceToHost);
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
    cudaFree(dev_u);
    cudaFree(dev_v);
    cudaFree(dev_p);
    cudaFree(dev_res);

    return 0;
}