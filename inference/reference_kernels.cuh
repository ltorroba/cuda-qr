#ifndef REFERENCE_KERNELS_CUH
#define REFERENCE_KERNELS_CUH

#include <cublas_v2.h>

// Kernel declarations (now with compute_ prefix)
__global__ void compute_QRx(const void* kernel_data, const float* x, float* output, 
                           int m, int n, int r);

__global__ void compute_QRX(const void* kernel_data, const float* X, float* output,
                           int m, int n, int r, int k);

// Launch function declarations (these are now regular functions, not kernels)
void* setup_QRX(const float* A, const float* B, int m, int n, int r);

void launch_QRx(const void* kernel_data, const float* x, float* output, 
                int m, int n, int r);

void launch_QRX(const void* kernel_data, const float* X, float* output,
                int m, int n, int r, int k);

// Existing cuBLAS wrapper functions
void* setup_ABX(const float* A, const float* B, int m, int n, int r);

void cublas_ABx(const void* kernel_data, const float* x, float* output,
                int m, int n, int r);

void cublas_ABX(const void* kernel_data, const float* X, float* output,
                int m, int n, int r, int k);

#endif // REFERENCE_KERNELS_CUH 