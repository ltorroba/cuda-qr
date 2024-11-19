#ifndef REFERENCE_KERNELS_CUH
#define REFERENCE_KERNELS_CUH

#include <cublas_v2.h>

// Kernel declarations (now with compute_ prefix)
__global__ void compute_QRx(const float* Q, const float* R, const float* x, float* output, 
                           int m, int n, int r);

__global__ void compute_QRX(const float* Q, const float* R, const float* X, float* output,
                           int m, int n, int r, int k);

// Launch function declarations (these are now regular functions, not kernels)
void launch_QRx(const float* Q, const float* R, const float* x, float* output, 
                int m, int n, int r);

void launch_QRX(const float* Q, const float* R, const float* X, float* output,
                int m, int n, int r, int k);

// Existing cuBLAS wrapper functions
void cublas_ABx(cublasHandle_t handle, 
                const float* A, const float* B, const float* x,
                float* output, float* temp,
                int m, int n, int r);

void cublas_ABX(cublasHandle_t handle,
                const float* A, const float* B, const float* X,
                float* output, float* temp,
                int m, int n, int r, int k);

#endif // REFERENCE_KERNELS_CUH 