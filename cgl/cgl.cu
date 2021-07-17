/*
Author: Jason He
Version: 1.0 20210521 Serial version.
Version: 2.0 20210523 MPI version using parallel fftw.
Version: 3.0 20210602 CUDA version using cufft. Use N*N block with 1 thread each.
Version: 3.1 20210603 Use more than 1 thread per block.
Version: 3.2 20210604 Included usage of shared memories. Didn't improve the speed.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cufft.h>
#include <curand.h>

/*
Compute problem 37.5.1 in textbook on a 2D grid. Terminal time T=10000.

Inputs:
    N: the size of the grid in both directions(should be evenly divisible by the number of processes)
    c1: double precision parameter
    c3: double precision parameter
    M: number of time steps
    s: long integer seed value(optional)

Outputs:
	print arguments including the seed
	output grid value into "CGL.out" that contain
		the data at t=100k, k=0,1,2,...,10.
	print run time and output it to runtime.dat

To run, E.g.
$ ./cgl 128 1.5 0.25 100000 12345

*/

const int threadsPerBlock = 256;//for NVIDIA TESLA K20c maximum is 1024
__device__ __constant__ int dev_N;
__device__ __constant__ double dev_dt;
__device__ __constant__ double dev_C1r;
__device__ __constant__ double dev_C1i;
__device__ __constant__ double dev_c3;

__global__ void cpy(cufftDoubleComplex* dev_A, cufftDoubleComplex* dev_A1){
    //this kernel copy the grid values from A1 to A
    int k = threadIdx.x + blockIdx.x*blockDim.x;
    if(k < dev_N*dev_N)
        dev_A[k] = dev_A1[k];
}

__global__ void spectrald(cufftDoubleComplex* dev_A2){
    //this kernel do the spectral derivative(laplacian)
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < dev_N*dev_N){
        int j = i/dev_N;
        int jj = (j <= dev_N/2 ? j : j-dev_N);//the mode index
        int k = i%dev_N;
        int kk = (k <= dev_N/2 ? k : k-dev_N);//the mode index
        int coeff = -(jj*jj + kk*kk);
        dev_A2[i].x *= coeff;
        dev_A2[i].y *= coeff;
    }
}

__global__ void RK4(cufftDoubleComplex* dev_A, cufftDoubleComplex* dev_A1,
                    cufftDoubleComplex* dev_A2, double tcoeff){
    //this kernel do RK4 steps
    __shared__ cufftDoubleComplex A[threadsPerBlock];
    __shared__ cufftDoubleComplex A1[threadsPerBlock];
    __shared__ cufftDoubleComplex A2[threadsPerBlock];
    /*maximum threadsPerBlock is 1024, so A,A1,A2 use up all the shared memories 
    on NVIDIA TESLA K20c in the maximum case.*/
    int kk = threadIdx.x + blockIdx.x*blockDim.x;//global index
    int k = threadIdx.x;//local index
    if(kk < dev_N*dev_N){
        A[k] = dev_A[kk];
        A1[k] = dev_A1[kk];
        A2[k] = dev_A2[kk];
        double A1A1 = A1[k].x*A1[k].x + A1[k].y*A1[k].y;
        double tmp = A1[k].x;//update the imag part first so store the real part temporarily
        A1[k].x = A[k].x + tcoeff*dev_dt*(A1[k].x
                                        + (dev_C1r*A2[k].x - dev_C1i*A2[k].y)/dev_N/dev_N
                                        - A1A1*(A1[k].x + dev_c3*A1[k].y));
        A1[k].y = A[k].y + tcoeff*dev_dt*(A1[k].y
                                        + (dev_C1r*A2[k].y + dev_C1i*A2[k].x)/dev_N/dev_N
                                        - A1A1*(A1[k].y - dev_c3*tmp));
        dev_A1[kk] = A1[k];
    }
}//rescale by 1/N^2 to correct the data after fft

int main(int argc, char* argv[])
{
#ifndef M_PI
    const double M_PI = 4.0*atan(1.0);
#endif

    //Choose gpu device
    cudaDeviceProp prop;
    int dev;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.multiProcessorCount = 13;
    cudaChooseDevice(&dev, &prop);
    cudaSetDevice(dev);

    //load input parameters
    int argi = 0;
    const int N = atol(argv[++argi]);
    double c1 = atof(argv[++argi]);
    double c3 = atof(argv[++argi]);
    int M = atoi(argv[++argi]);
    long int seed;
    if (argi < argc-1){
        seed = atol(argv[++argi]);
    }
    else{
        seed = (long int)time(NULL);
    }
    srand48(seed);

    printf( "N = %d\nc1 = %lf\nc3 = %lf\nM = %d\n"
        "Starting seed = %ld\n", N, c1, c3, M, seed);
   
    //parameters for calculation. refer to textbook Eq(37.15)
    double dt = 10000.0/M;//terminal time = 10000
    double C1r = 1.0/64/64;//L = 128pi
    double C1i = C1r*c1;
    cudaMemcpyToSymbol(dev_dt, &dt, sizeof(double));
    cudaMemcpyToSymbol(dev_C1r, &C1r, sizeof(double));
    cudaMemcpyToSymbol(dev_C1i, &C1i, sizeof(double));
    cudaMemcpyToSymbol(dev_c3, &c3, sizeof(double));
    //parameters for the blocks and grids
    cudaMemcpyToSymbol(dev_N, &N, sizeof(double));
    const int blocksPerGrid = N*N/threadsPerBlock + ((N*N)%threadsPerBlock > 0 ? 1 : 0);
    
    //initialize value of the grid on host
    /*because we need to output the initial data, might as well
    initialize it on the host, output to file, and then transfer the data to the device.*/
    cufftDoubleComplex *A;
    A = (cufftDoubleComplex*)malloc(N*N*sizeof(cufftDoubleComplex));
    for (int i = 0; i < N*N; i++){
        A[i].x = 3*drand48() -1.5;
        A[i].y = 3*drand48() -1.5;
    }
    
    //initialize value of the grid on device
    cufftDoubleComplex *dev_A, *dev_A1, *dev_A2;
    cudaMalloc((void**)&dev_A, N*N*sizeof(cufftDoubleComplex));
    cudaMalloc((void**)&dev_A1, N*N*sizeof(cufftDoubleComplex));
    cudaMalloc((void**)&dev_A2, N*N*sizeof(cufftDoubleComplex));
    cudaMemcpy(dev_A1, A, N*N*sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
    cpy<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_A1);
   
    //file output the initial value
    FILE* file = fopen("CGL.out","w");
    fwrite(A, sizeof(cufftDoubleComplex), N*N, file);
    printf("Saved output at t=0\n");
   
    //cufft plans
    cufftHandle plan;
    cufftPlan2d(&plan, N, N, CUFFT_Z2Z);

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    //main loop
    for(int step=0; step<M; ++step){
        //first step in RK4
        cufftExecZ2Z(plan, dev_A1, dev_A2, CUFFT_FORWARD);
        spectrald<<<blocksPerGrid, threadsPerBlock>>>(dev_A2);
        cufftExecZ2Z(plan, dev_A2, dev_A2, CUFFT_INVERSE);
        RK4<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_A1, dev_A2, 0.25);
        
        //second step in RK4
        cufftExecZ2Z(plan, dev_A1, dev_A2, CUFFT_FORWARD);
        spectrald<<<blocksPerGrid, threadsPerBlock>>>(dev_A2);
        cufftExecZ2Z(plan, dev_A2, dev_A2, CUFFT_INVERSE);
        RK4<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_A1, dev_A2, 1.0/3);
        
        //third step in RK4
        cufftExecZ2Z(plan, dev_A1, dev_A2, CUFFT_FORWARD);
        spectrald<<<blocksPerGrid, threadsPerBlock>>>(dev_A2);
        cufftExecZ2Z(plan, dev_A2, dev_A2, CUFFT_INVERSE);
        RK4<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_A1, dev_A2, 0.5);
        
        //fourth step in RK4
        cufftExecZ2Z(plan, dev_A1, dev_A2, CUFFT_FORWARD);
        spectrald<<<blocksPerGrid, threadsPerBlock>>>(dev_A2);
        cufftExecZ2Z(plan, dev_A2, dev_A2, CUFFT_INVERSE);
        RK4<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_A1, dev_A2, 1.0);
        
        //store the final value of this time step to A
        cpy<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_A1);
        
        //output to files
        if((step+1)%(M/10) == 0)
        {
            cudaMemcpy(A, dev_A, N*N*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
            fwrite(A, sizeof(cufftDoubleComplex), N*N, file);
            printf("Saved output at t=%d\n", (step+1)/10);
        }
    }//main loop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //close files and free memories
    fclose(file);
    cufftDestroy(plan);
    cudaFree(dev_A);
    cudaFree(dev_A1);
    cudaFree(dev_A2);
    free(A);

    //print runtime and output to the file
    FILE* tfile = fopen("runtime.dat", "a");
    fprintf(tfile, "%d %lf\n", N, elapsedTime);
    printf("Time: %gms\n", elapsedTime);
    fclose(tfile);

    return 0;
}