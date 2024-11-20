#include "reference_kernels.cuh"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "cuda_utils.cuh"

void* setup_QRX(const float* A, const float* B, int m, int n, int r) {
    // Allocate memory for Q and R
    size_t total_size = (m * n + n * n) * sizeof(float);
    void* data;
    CHECK_CUDA(cudaMalloc(&data, total_size));
    
    // First compute AB
    float* AB;
    CHECK_CUDA(cudaMalloc(&AB, m * n * sizeof(float)));
    
    // Use cuBLAS to compute AB
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, r,              // dimensions
                &alpha,
                A, m,                 // matrix A (m x r)
                B, r,                 // matrix B (r x n)
                &beta,
                AB, m));              // output AB (m x n)
    
    // Setup cuSolver for QR
    cusolverDnHandle_t solver_handle;
    CHECK_CUSOLVER(cusolverDnCreate(&solver_handle));
    
    // Query working space for QR
    int lwork;
    float* tau;
    CHECK_CUDA(cudaMalloc(&tau, n * sizeof(float)));
    CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(solver_handle, m, n, AB, m, &lwork));
    
    // Allocate working space
    float* workspace;
    CHECK_CUDA(cudaMalloc(&workspace, lwork * sizeof(float)));
    int* devInfo;
    CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
    
    // Compute QR factorization in-place in AB
    CHECK_CUSOLVER(cusolverDnSgeqrf(solver_handle, m, n,
                     AB, m,
                     tau, workspace, lwork,
                     devInfo));
    
    // Copy R from upper triangular part of AB
    float* R = static_cast<float*>(data) + m * n;  // R starts after Q
    CHECK_CUDA(cudaMemset(R, 0, n * n * sizeof(float)));
    for(int j = 0; j < n; j++) {
        CHECK_CUDA(cudaMemcpy(R + j * n, AB + j * m, (j + 1) * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    // Compute explicit Q matrix
    CHECK_CUSOLVER(cusolverDnSorgqr(solver_handle, m, n, n,
                     AB, m,
                     tau, workspace, lwork,
                     devInfo));
    
    // Copy Q to output data
    float* Q = static_cast<float*>(data);  // Q comes first in data
    CHECK_CUDA(cudaMemcpy(Q, AB, m * n * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Cleanup
    CHECK_CUDA(cudaFree(AB));
    CHECK_CUDA(cudaFree(tau));
    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUSOLVER(cusolverDnDestroy(solver_handle));
    CHECK_CUBLAS(cublasDestroy(handle));
    
    return data;
}

__global__ void compute_QRx(const void* kernel_data, const float* x, float* output, int m, int n, int r) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    const float* Q = static_cast<const float*>(kernel_data);
    const float* R = Q + m * n;

    float sum = 0.0f;
    for (int j = 0; j < r; j++) {
        float QR_row_j = 0.0f;
        for (int k = 0; k < n; k++) {
            QR_row_j += Q[row + k * m] * R[k + j * n];
            printf("Q[%d] = %f\n", row + k * m, Q[row + k * m]);
            printf("R[%d] = %f\n", k + j * n, R[k + j * n]);
        }
        printf("x[%d] = %f\n", j, x[j]);
        sum += QR_row_j * x[j];
    }
    output[row] = sum;
}

__global__ void compute_QRX(const void* kernel_data, const float* X, float* output,
                           int m, int n, int r, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= m || col >= k) return;

    const float* Q = static_cast<const float*>(kernel_data);
    const float* R = Q + m * n;
    
    float sum = 0.0f;
    for (int j = 0; j < r; j++) {
        float QR_row_j = 0.0f;
        for (int i = 0; i < n; i++) {
            QR_row_j += Q[row + i * m] * R[i + j * n];
        }
        sum += QR_row_j * X[j + col * r];
    }
    output[row + col * m] = sum;
}

// Launch function implementations
void launch_QRx(const void* kernel_data, const float* x, float* output, 
                int m, int n, int r) {
    // Choose block size and compute grid size
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    compute_QRx<<<grid, block>>>(kernel_data, x, output, m, n, r);
    CHECK_KERNEL();
    CHECK_CUDA(cudaDeviceSynchronize());
}

void launch_QRX(const void* kernel_data, const float* X, float* output,
                int m, int n, int r, int k) {
    // Choose block dimensions
    const int BLOCK_DIM_X = 16;
    const int BLOCK_DIM_Y = 16;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((m + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (k + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    
    // Launch kernel
    compute_QRX<<<grid, block>>>(kernel_data, X, output, m, n, r, k);
    CHECK_KERNEL();
    CHECK_CUDA(cudaDeviceSynchronize());
}

void* setup_ABX(const float* A, const float* B, int m, int n, int r) {
    size_t total_size = (m * r + r * n) * sizeof(float);
    float* data;
    CHECK_CUDA(cudaMalloc(&data, total_size));

    // Copy A and B to data
    CHECK_CUDA(cudaMemcpy(data, A, m * r * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(data + m * r, B, r * n * sizeof(float), cudaMemcpyDeviceToDevice));

    return data;
}

void cublas_ABx(const void* kernel_data, const float* x, float* output,
                int m, int n, int r) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float* temp;
    CHECK_CUDA(cudaMalloc(&temp, n * sizeof(float)));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const float* A = static_cast<const float*>(kernel_data);
    const float* B = A + m * r;

    // First compute Bx = temp
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N,
                n, r,                  // dimensions
                &alpha,
                B, n,                  // matrix B (n x r)
                x, 1,                  // vector x (r x 1)
                &beta,
                temp, 1));            // output temp (n x 1)

    // Then compute A(Bx) = output
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N,
                m, n,                  // dimensions
                &alpha,
                A, m,                  // matrix A (m x n)
                temp, 1,               // vector temp (n x 1)
                &beta,
                output, 1));          // output (m x 1)

    // Cleanup
    CHECK_CUDA(cudaFree(temp));
    CHECK_CUBLAS(cublasDestroy(handle));
}

void cublas_ABX(const void* kernel_data, const float* X, float* output,
                int m, int n, int r, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    float* temp;
    CHECK_CUDA(cudaMalloc(&temp, n * sizeof(float)));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float* A = static_cast<const float*>(kernel_data);
    const float* B = A + m * r;

    // First compute BX = temp
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, k, r,              // dimensions
                &alpha,
                B, n,                  // matrix B (n x r)
                X, r,                  // matrix X (r x k)
                &beta,
                temp, n));             // output temp (n x k)

    // Then compute A(BX) = output
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, k, n,              // dimensions
                &alpha,
                A, m,                  // matrix A (m x n)
                temp, n,               // matrix temp (n x k)
                &beta,
                output, m));           // output (m x k)
} 