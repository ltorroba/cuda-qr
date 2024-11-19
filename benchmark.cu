#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <chrono>
#include <cusolverDn.h>

#include "reference_kernels.cuh"

// Add this before main()
template<typename F>
double benchmark_kernel(F func, int num_trials = 1) {
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_trials; i++) {
        func();
        cudaDeviceSynchronize();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)num_trials;
}

// Helper function to generate random low-rank matrix
std::tuple<float*, float*, float*> generateRandomLowRankMatrix(
    cublasHandle_t handle, curandGenerator_t gen, int m, int n, int rank) {
    // Generate A = UV^T where U is m x rank and V is n x rank
    float* d_U;
    float* d_V;
    float* d_A;
    cudaMalloc(&d_U, m * rank * sizeof(float));
    cudaMalloc(&d_V, n * rank * sizeof(float));
    cudaMalloc(&d_A, m * n * sizeof(float));

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

    return {d_A, d_U, d_V};
}

struct CompactQR {
    float* d_A;      // Compact form (m x n)
    float* d_tau;    // Householder scalars (n)
    int m;           // Number of rows
    int n;           // Number of columns
    
    // Destructor to handle cleanup
    ~CompactQR() {
        if (d_A) cudaFree(d_A);
        if (d_tau) cudaFree(d_tau);
    }
    
    // Prevent copying
    CompactQR(const CompactQR&) = delete;
    CompactQR& operator=(const CompactQR&) = delete;
    
    // Allow moving
    CompactQR(CompactQR&& other) noexcept 
        : d_A(other.d_A), d_tau(other.d_tau), m(other.m), n(other.n) {
        other.d_A = nullptr;
        other.d_tau = nullptr;
    }
    
    // Constructor
    CompactQR(float* a, float* tau, int rows, int cols) 
        : d_A(a), d_tau(tau), m(rows), n(cols) {}
};

std::tuple<float*, float*, CompactQR> computeQR(cusolverDnHandle_t solver_handle, 
                                               const float* d_input, int m, int n) {
    // Allocate output matrices
    float* d_Q;
    float* d_R;
    float* d_A;  // For compact form
    cudaMalloc(&d_Q, m * n * sizeof(float));
    cudaMalloc(&d_R, n * n * sizeof(float));
    cudaMalloc(&d_A, m * n * sizeof(float));
    
    // Copy input to compact form
    cudaMemcpy(d_A, d_input, m * n * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Allocate workspace for QR decomposition
    int work_size = 0;
    cusolverDnSgeqrf_bufferSize(solver_handle, m, n, d_A, m, &work_size);
    
    float* d_work;
    float* d_tau;
    int* d_info;
    cudaMalloc(&d_work, work_size * sizeof(float));
    cudaMalloc(&d_tau, n * sizeof(float));
    cudaMalloc(&d_info, sizeof(int));

    // Compute QR factorization (A = QR) in compact form
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

    // Cleanup temporary workspace
    cudaFree(d_work);
    cudaFree(d_info);
    
    // Create CompactQR struct (transfers ownership of d_A and d_tau)
    CompactQR compact(d_A, d_tau, m, n);
    
    return std::make_tuple(d_Q, d_R, std::move(compact));
}

int main(int argc, char **argv) {
    const int num_trials = 100;
    const int m = 1024;  // matrix height
    const int n = 512;   // matrix width
    const int r = 32;    // rank
    const int batch_size = 10;  // for matrix X

    // Initialize CUDA resources
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    // Allocate device memory
    float *d_Q, *d_R, *d_x, *d_X, *d_output, *d_output_matrix;
    cudaMalloc(&d_Q, m * n * sizeof(float));
    cudaMalloc(&d_R, n * n * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_X, n * batch_size * sizeof(float));
    cudaMalloc(&d_output, m * sizeof(float));
    cudaMalloc(&d_output_matrix, m * batch_size * sizeof(float));

    float *d_temp;
    cudaMalloc(&d_temp, n * sizeof(float));

    float *d_output_ref, *d_output_matrix_ref;
    cudaMalloc(&d_output_ref, m * sizeof(float));
    cudaMalloc(&d_output_matrix_ref, m * batch_size * sizeof(float));

    // Benchmark vector multiplication
    float total_time_vector = 0.0f;
    float total_time_vector_ref = 0.0f;
    float max_error_vector = 0.0f;

    std::cout << "Benchmarking vector multiplication...\n";
    for (int trial = 0; trial < num_trials; trial++) {
        // Generate random low-rank matrix
        auto [d_A, d_U, d_V] = generateRandomLowRankMatrix(handle, gen, m, n, r);
        auto [d_Q, d_R, compact] = computeQR(solver_handle, d_A, m, n);
        
        // Generate random vector x
        curandGenerateNormal(gen, d_x, r, 0.0f, 1.0f);
        
        // Compute reference result
        cublas_ABx(handle, d_Q, d_R, d_x, d_output_ref, d_temp, m, n, r);
        
        // Test each implementation
        for (size_t i = 0; i < kernels.size(); i++) {
            auto& kernel = kernels[i];
            auto& results = vector_results[i];
            
            // Time the kernel
            auto kernel_time = benchmark_kernel([&]() {
                kernel.qrx_kernel(d_Q, d_R, d_x, d_output, m, n, r);
            });
            results.total_time += kernel_time;
            
            // Compare with reference
            thrust::device_vector<float> custom_result(m);
            thrust::device_vector<float> ref_result(m);
            cudaMemcpy(custom_result.data().get(), d_output, m * sizeof(float), 
                      cudaMemcpyDeviceToDevice);
            cudaMemcpy(ref_result.data().get(), d_output_ref, m * sizeof(float), 
                      cudaMemcpyDeviceToDevice);
            
            for (int j = 0; j < m; j++) {
                float diff = std::abs(custom_result[j] - ref_result[j]);
                results.max_error = std::max(results.max_error, diff);
            }
        }
        
        // Cleanup
        cudaFree(d_A);
        cudaFree(d_U);
        cudaFree(d_V);
    }
    
    // Matrix multiplication benchmark
    std::vector<KernelResults> matrix_results(kernels.size());
    
    for (int trial = 0; trial < num_trials; trial++) {
        // Generate random low-rank matrix
        auto [d_A, d_U, d_V] = generateRandomLowRankMatrix(handle, gen, m, n, r);
        auto [d_Q, d_R, compact] = computeQR(solver_handle, d_A, m, n);
        
        // Generate random matrix X
        curandGenerateNormal(gen, d_X, n * batch_size, 0.0f, 1.0f);
        
        // Compute reference result
        cublas_ABX(handle, d_Q, d_R, d_X, d_output_matrix_ref, d_temp, m, n, r, batch_size);
        
        // Test each implementation
        for (size_t i = 0; i < kernels.size(); i++) {
            auto& kernel = kernels[i];
            auto& results = matrix_results[i];
            
            // Time the kernel
            auto kernel_time = benchmark_kernel([&]() {
                kernel.qrX_kernel(d_Q, d_R, d_X, d_output_matrix, m, n, r, batch_size);
            });
            results.total_time += kernel_time;
            
            // Compare with reference
            thrust::device_vector<float> custom_result(m * batch_size);
            thrust::device_vector<float> ref_result(m * batch_size);
            cudaMemcpy(custom_result.data().get(), d_output_matrix, m * batch_size * sizeof(float), 
                      cudaMemcpyDeviceToDevice);
            cudaMemcpy(ref_result.data().get(), d_output_matrix_ref, m * batch_size * sizeof(float), 
                      cudaMemcpyDeviceToDevice);
            
            for (int j = 0; j < m * batch_size; j++) {
                float diff = std::abs(custom_result[j] - ref_result[j]);
                results.max_error = std::max(results.max_error, diff);
            }
        }
        
        // Cleanup
        cudaFree(d_A);
        cudaFree(d_U);
        cudaFree(d_V);
    }
    
    // Print results
    std::cout << "\nVector Operation Results:\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << std::setw(20) << "Implementation" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Max Error" << "\n";
    std::cout << std::string(50, '-') << "\n";
    
    for (size_t i = 0; i < kernels.size(); i++) {
        std::cout << std::setw(20) << kernels[i].name 
                  << std::setw(15) << (vector_results[i].total_time / num_trials) / 1000
                  << std::setw(15) << vector_results[i].max_error << "\n";
    }
    
    std::cout << "\nMatrix Operation Results:\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << std::setw(20) << "Implementation" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Max Error" << "\n";
    std::cout << std::string(50, '-') << "\n";
    
    for (size_t i = 0; i < kernels.size(); i++) {
        std::cout << std::setw(20) << kernels[i].name 
                  << std::setw(15) << (matrix_results[i].total_time / num_trials) / 1000
                  << std::setw(15) << matrix_results[i].max_error << "\n";
    }
    
    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_R);
    cudaFree(d_x);
    cudaFree(d_X);
    cudaFree(d_output);
    cudaFree(d_output_matrix);
    cudaFree(d_temp);
    cudaFree(d_output_ref);
    cudaFree(d_output_matrix_ref);
    cudaFree(d_temp_vector);
    cudaFree(d_temp_matrix);
    cublasDestroy(handle);
    curandDestroyGenerator(gen);
    cusolverDnDestroy(solver_handle);

    return 0;
}