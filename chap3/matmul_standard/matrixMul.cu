#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <curand.h>

// https://stackoverflow.com/a/76073425
#define checkCudaErrors(err)                   \
    do                                         \
    {                                          \
        cuda_check((err), __FILE__, __LINE__); \
    } while (false)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

// checkCudaErrors adapted for curand
#define checkCurandErrors(err)                   \
    do                                           \
    {                                            \
        curand_check((err), __FILE__, __LINE__); \
    } while (false)
inline void curand_check(curandStatus_t error_code, const char *file, int line)
{
    if (error_code != CURAND_STATUS_SUCCESS)
    {
        fprintf(stderr, "CURAND Error %d In file '%s' on line %d\n", error_code, file, line);
        fflush(stderr);
        exit(error_code);
    }
}

__global__ void matrixMulKernel(float *M, float *N, float *P, dim3 MDim, dim3 NDim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < MDim.y && col < NDim.x)
    {
        float pValue = 0;
        for (int k = 0; k < NDim.y; ++k)
        {
            pValue += M[row * MDim.x + k] * N[k * NDim.x + col];
        }
        P[row * NDim.x + col] = pValue;
    }
}

__global__ void floatArrScalingKernel(float *arr, float scale, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        arr[i] *= scale;
    }
}

int matrixMultiply(int blockSize, dim3 ADim, dim3 BDim)
{

    // Allocate host memory for matrices A and B
    unsigned int sizeA = ADim.y * ADim.x;
    unsigned int memSizeA = sizeof(float) * sizeA;
    float *h_A;
    checkCudaErrors(cudaMallocHost(&h_A, memSizeA));

    unsigned int sizeB = BDim.y * BDim.x;
    unsigned int memSizeB = sizeof(float) * sizeB;
    float *h_B;
    checkCudaErrors(cudaMallocHost(&h_B, memSizeB));

    unsigned int sizeC = ADim.y * BDim.x;
    unsigned int memSizeC = sizeof(float) * sizeC;
    float *h_C;
    checkCudaErrors(cudaMallocHost(&h_C, memSizeC));

    // Allocate device memory
    float *d_A;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), memSizeA));
    float *d_B;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), memSizeB));
    float *d_C;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), memSizeC));

    // generate random floats for matrix A and B
    curandGenerator_t gen;
    checkCurandErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    checkCurandErrors(curandSetPseudoRandomGeneratorSeed(gen, 42ULL));

    checkCurandErrors(curandGenerateUniform(gen, d_A, sizeA));
    checkCurandErrors(curandGenerateUniform(gen, d_B, sizeB));

    // scale random floats (for fun)
    float scale = 10.0f;
    floatArrScalingKernel<<<(sizeA + blockSize - 1) / blockSize, blockSize>>>(d_A, scale, sizeA);
    checkCudaErrors(cudaGetLastError());
    floatArrScalingKernel<<<(sizeB + blockSize - 1) / blockSize, blockSize>>>(d_B, scale, sizeB);
    checkCudaErrors(cudaGetLastError());

    // matrix multiplication
    dim3 threads(blockSize, blockSize);
    dim3 grid((BDim.x + threads.x - 1) / threads.x, (ADim.y + threads.y - 1) / threads.y);
    matrixMulKernel<<<grid, threads>>>(d_A, d_B, d_C, ADim, BDim);

    // copy d_C from device back to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, memSizeC, cudaMemcpyDeviceToHost));

    // print subsample of results
    int printSize = 5;
    printf("\nC (first %i): ", printSize);
    for (int i = 0; i < printSize; i++)
    {
        printf("%f ", h_C[i]);
    }

    // free memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    return EXIT_SUCCESS;
}

int main()
{
    // Part of the template is from here:
    // https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/matrixMul/matrixMul.cu
    printf("Starting...\n");

    int blockSize = 32;

    int matrixARows = 4096;
    int matrixAColumns = 8192;
    int matrixBRows = 8192;
    int matrixBColumns = 3072;

    // Small test matrix
    // Equivalent Wolfram alpha matrix
    // {{0.700209,1.601456},{4.204961,5.763708},{1.577025,4.019262},{5.076846,1.087533}}.{{9.662183,5.387602,1.602132},{9.520704,6.733531,4.391682}}
    // int matrixARows = 4;    // 4096;
    // int matrixAColumns = 2; // 8192;
    // int matrixBRows = 2;    // 8192;
    // int matrixBColumns = 3; // 3072;

    dim3 ADim(matrixAColumns, matrixARows);
    dim3 BDim(matrixBColumns, matrixBRows);

    if (matrixAColumns != matrixBRows)
    {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               matrixAColumns, matrixBRows);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", matrixARows, matrixAColumns,
           matrixBRows, matrixBColumns);

    checkCudaErrors(cudaProfilerStart());
    int res = matrixMultiply(blockSize, ADim, BDim);
    checkCudaErrors(cudaProfilerStop());

    exit(res);
}
