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
        const char* error_msg; \
        switch (status) { \
            case CUSOLVER_STATUS_NOT_INITIALIZED: \
                error_msg = "CUSOLVER_STATUS_NOT_INITIALIZED: Library not initialized"; \
                break; \
            case CUSOLVER_STATUS_ALLOC_FAILED: \
                error_msg = "CUSOLVER_STATUS_ALLOC_FAILED: Resource allocation failed"; \
                break; \
            case CUSOLVER_STATUS_INVALID_VALUE: \
                error_msg = "CUSOLVER_STATUS_INVALID_VALUE: Invalid value passed"; \
                break; \
            case CUSOLVER_STATUS_ARCH_MISMATCH: \
                error_msg = "CUSOLVER_STATUS_ARCH_MISMATCH: Architecture mismatch"; \
                break; \
            case CUSOLVER_STATUS_EXECUTION_FAILED: \
                error_msg = "CUSOLVER_STATUS_EXECUTION_FAILED: Execution failed"; \
                break; \
            case CUSOLVER_STATUS_INTERNAL_ERROR: \
                error_msg = "CUSOLVER_STATUS_INTERNAL_ERROR: Internal operation failed"; \
                break; \
            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: \
                error_msg = "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: Matrix type not supported"; \
                break; \
            default: \
                error_msg = "Unknown cuSOLVER error"; \
        } \
        printf("cuSOLVER Error: %s:%d\nStatus: %d\nMessage: %s\n", \
               __FILE__, __LINE__, status, error_msg); \
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

inline void convert_matrix_major(const float* input_d,
                         float* output_d,
                         int m,      // rows
                         int n,      // cols
                         bool to_column_major = true) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (to_column_major) {
        // Converting row-major to column-major
        CHECK_CUBLAS(cublasSgeam(handle,
                                CUBLAS_OP_T,    // transpose
                                CUBLAS_OP_N,    // no-op
                                m, n,           // output dimensions
                                &alpha,
                                input_d,        // input (viewed as n x m by cuBLAS)
                                n,              // leading dimension of input
                                &beta,
                                nullptr,        // no B matrix
                                m,              // ldb (unused)
                                output_d,       // output
                                m));            // leading dimension of output
    } else {
        // Converting column-major to row-major
        CHECK_CUBLAS(cublasSgeam(handle,
                                CUBLAS_OP_T,    // transpose
                                CUBLAS_OP_N,    // no-op
                                n, m,           // output dimensions (swapped)
                                &alpha,
                                input_d,        // input
                                m,              // leading dimension of input
                                &beta,
                                nullptr,        // no B matrix
                                n,              // ldb (unused)
                                output_d,       // output
                                n));            // leading dimension of output
    }

    CHECK_CUBLAS(cublasDestroy(handle));
}
