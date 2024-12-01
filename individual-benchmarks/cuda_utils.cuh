#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <curand.h>
#include <stdio.h>

// Macro for checking CUDA errors
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

// Macro for checking cuBLAS errors
#define CHECK_CUBLAS(call) { \
    const cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error: %s:%d, status: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

// Macro for checking cuSOLVER errors
#define CHECK_CUSOLVER(call) { \
    const cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        printf("cuSOLVER Error: %s:%d, status: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

// Macro for checking kernel launches
#define CHECK_KERNEL() { \
    const cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        printf("Kernel Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} 

#define CHECK_CURAND(call) { \
    const curandStatus_t status = call; \
    if (status != CURAND_STATUS_SUCCESS) { \
        printf("cuRAND Error: %s:%d, status: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}