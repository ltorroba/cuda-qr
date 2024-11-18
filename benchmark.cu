#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <chrono>
#include <cusolverDn.h>
#include "reference_kernels.cuh"

// Helper function to generate random low-rank matrix
void generateRandomLowRankMatrix(cublasHandle_t handle, curandGenerator_t gen,
                               float* d_A, int m, int n, int rank) {
    // Generate A = UV^T where U is m x rank and V is n x rank
    float* d_U;
    float* d_V;
    cudaMalloc(&d_U, m * rank * sizeof(float));
    cudaMalloc(&d_V, n * rank * sizeof(float));

    // Generate random U and V
    curandGenerateNormal(gen, d_U, m * rank, 0.0f, 1.0f);
    curandGenerateNormal(gen, d_V, n * rank, 0.0f, 1.0f);

    // Compute A = U * V^T
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                m, n, rank,
                &alpha,
                d_U, m,
                d_V, n,
                &beta,
                d_A, m);

    cudaFree(d_U);
    cudaFree(d_V);
}

void computeQR(cusolverDnHandle_t solver_handle, float* d_A, float* d_Q, float* d_R, 
               int m, int n) {
    // Allocate workspace for QR decomposition
    int work_size = 0;
    cusolverDnSgeqrf_bufferSize(solver_handle, m, n, d_A, m, &work_size);
    
    float* d_work;
    float* d_tau;
    int* d_info;
    cudaMalloc(&d_work, work_size * sizeof(float));
    cudaMalloc(&d_tau, n * sizeof(float));
    cudaMalloc(&d_info, sizeof(int));

    // Compute QR factorization (A = QR)
    cusolverDnSgeqrf(solver_handle, m, n, d_A, m, d_tau, d_work, work_size, d_info);

    // Copy upper triangular part to R
    cudaMemset(d_R, 0, n * n * sizeof(float));
    for(int i = 0; i < n; i++) {
        cudaMemcpy(&d_R[i * n], &d_A[i * m], (i + 1) * sizeof(float), 
                  cudaMemcpyDeviceToDevice);
    }

    // Compute Q explicitly
    cudaMemcpy(d_Q, d_A, m * n * sizeof(float), cudaMemcpyDeviceToDevice);
    cusolverDnSorgqr(solver_handle, m, n, n, d_Q, m, d_tau, d_work, work_size, d_info);

    // Cleanup
    cudaFree(d_work);
    cudaFree(d_tau);
    cudaFree(d_info);
}

int main(int argc, char **argv) {
    const int num_trials = 1000;
    const int m = 1024;  // matrix height
    const int n = 512;   // matrix width
    const int r = 32;    // rank
    const int batch_size = 100;  // for matrix X

    // Initialize CUDA resources
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    // Allocate device memory
    float *d_A, *d_Q, *d_R, *d_x, *d_X, *d_output, *d_output_matrix;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_Q, m * n * sizeof(float));
    cudaMalloc(&d_R, n * n * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_X, n * batch_size * sizeof(float));
    cudaMalloc(&d_output, m * sizeof(float));
    cudaMalloc(&d_output_matrix, m * batch_size * sizeof(float));

    // Benchmark vector multiplication
    double total_time_vector = 0.0;
    for (int trial = 0; trial < num_trials; trial++) {
        // Generate random low-rank matrix
        generateRandomLowRankMatrix(handle, gen, d_A, m, n, r);
        
        // Compute QR decomposition
        computeQR(solver_handle, d_A, d_Q, d_R, m, n);
        
        // Generate random vector x
        curandGenerateNormal(gen, d_x, n, 0.0f, 1.0f);

        // Time the vector multiplication
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        
        // Launch your kernel (you'll need to set appropriate grid/block dimensions)
        launch_QRx<<<(m + 255) / 256, 256>>>(d_Q, d_R, d_x, d_output, m, n, r);
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        total_time_vector += std::chrono::duration<double>(end - start).count();
    }

    // Benchmark matrix multiplication
    double total_time_matrix = 0.0;
    for (int trial = 0; trial < num_trials; trial++) {
        // Generate random low-rank matrix
        generateRandomLowRankMatrix(handle, gen, d_A, m, n, r);
        
        // Compute QR decomposition
        computeQR(solver_handle, d_A, d_Q, d_R, m, n);
        
        // Generate random matrix X
        curandGenerateNormal(gen, d_X, n * batch_size, 0.0f, 1.0f);

        // Time the matrix multiplication
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        
        launch_QRX<<<dim3((m + 15) / 16, (batch_size + 15) / 16), dim3(16, 16)>>>(
            d_Q, d_R, d_X, d_output_matrix, m, n, r, batch_size);
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        total_time_matrix += std::chrono::duration<double>(end - start).count();
    }

    // Print results
    std::cout << "Average time for QRx: " << (total_time_vector / num_trials) * 1000 << " ms\n";
    std::cout << "Average time for QRX: " << (total_time_matrix / num_trials) * 1000 << " ms\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_Q);
    cudaFree(d_R);
    cudaFree(d_x);
    cudaFree(d_X);
    cudaFree(d_output);
    cudaFree(d_output_matrix);
    cublasDestroy(handle);
    curandDestroyGenerator(gen);
    cusolverDnDestroy(solver_handle);

    return 0;
}
