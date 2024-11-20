#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <functional>
#include <cusolverDn.h>

#include "reference_kernels.cuh"
#include "cuda_utils.cuh"

template<typename F>
double benchmark_kernel(F func, int num_trials = 1) {
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_trials; i++) {
        func();
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double)num_trials;
}

// Helper function to generate random low-rank matrix
std::tuple<float*, float*, float*> generateRandomLowRankMatrix(
    cublasHandle_t handle, curandGenerator_t gen, int m, int n, int rank) {
    // Generate A = UV^T where U is m x rank and V is n x rank
    float* d_A;
    float* d_B;
    float* d_AB;
    CHECK_CUDA(cudaMalloc(&d_A, m * rank * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, n * rank * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_AB, m * n * sizeof(float)));

    // Generate random U and V
    CHECK_CURAND(curandGenerateNormal(gen, d_A, m * rank, 0.0f, 1.0f));
    CHECK_CURAND(curandGenerateNormal(gen, d_B, n * rank, 0.0f, 1.0f));

    // Compute A = U * V^T
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                m, n, rank,
                &alpha,
                d_A, m,
                d_B, n,
                &beta,
                d_AB, m));

    return {d_AB, d_A, d_B};
}

struct KernelResults {
    float total_time = 0.0f;
    float max_error = 0.0f;
};

struct KernelPair {
    std::string name;
    std::function<void*(const float*, const float*, int, int, int)> setup;
    std::function<void(const void*, const float*, float*, int, int, int)> qrx_kernel;
    std::function<void(const void*, const float*, float*, int, int, int, int)> qrX_kernel;
    
    KernelPair(const std::string& n,
               std::function<void*(const float*, const float*, int, int, int)> setup_fn,
               std::function<void(const void*, const float*, float*, int, int, int)> x_kernel,
               std::function<void(const void*, const float*, float*, int, int, int, int)> X_kernel)
        : name(n), setup(setup_fn), qrx_kernel(x_kernel), qrX_kernel(X_kernel) {}
};

int main(int argc, char **argv) {
    const int num_trials = 1;
    const int m = 20;  // matrix height
    const int n = 12;   // matrix width
    const int r = 1;    // rank
    const int batch_size = 2;  // for matrix X

    // Initialize CUDA resources
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    cusolverDnHandle_t solver_handle;
    CHECK_CUSOLVER(cusolverDnCreate(&solver_handle));

    // Allocate device memory
    float *d_x, *d_X, *d_output, *d_output_matrix;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_X, n * batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_matrix, m * batch_size * sizeof(float)));

    float *d_output_ref, *d_output_matrix_ref;
    CHECK_CUDA(cudaMalloc(&d_output_ref, m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_matrix_ref, m * batch_size * sizeof(float)));

    // Allocate temp buffers for cuBLAS
    float* d_temp_vector;
    float* d_temp_matrix;
    CHECK_CUDA(cudaMalloc(&d_temp_vector, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_temp_matrix, n * batch_size * sizeof(float)));

    // Define all kernel implementations
    std::vector<KernelPair> kernels = {
        KernelPair("Custom implementation",
            setup_QRX,
            launch_QRx, 
            launch_QRX
        )
        // KernelPair("Baseline (cuBLAS)",
        //     setup_ABX,
        //     cublas_ABx,
        //     cublas_ABX
        // )
    };

    // Benchmark vector multiplication
    std::cout << "Benchmarking vector multiplication...\n";
    std::vector<KernelResults> vector_results(kernels.size());

    for (int trial = 0; trial < num_trials; trial++) {
        // Generate random low-rank matrix
        auto [d_AB, d_A, d_B] = generateRandomLowRankMatrix(handle, gen, m, n, r);
        
        // Generate random vector x
        CHECK_CURAND(curandGenerateNormal(gen, d_x, n, 0.0f, 1.0f));
        
        // Compute reference result
        auto ref_kernel_data = setup_ABX(d_A, d_B, m, n, r);
        cublas_ABx(ref_kernel_data, d_x, d_output_ref, m, n, r);
        
        // Test each implementation
        for (size_t i = 0; i < kernels.size(); i++) {
            auto& kernel = kernels[i];
            auto& results = vector_results[i];
            
            // Time the kernel
            auto kernel_data = kernel.setup(d_A, d_B, m, n, r);
            auto kernel_time = benchmark_kernel([&]() {
                kernel.qrx_kernel(kernel_data, d_x, d_output, m, n, r);
            });
            results.total_time += kernel_time;
            
            // Compare with reference
            thrust::device_ptr<float> custom_result = thrust::device_pointer_cast(d_output);
            thrust::device_ptr<float> ref_result = thrust::device_pointer_cast(d_output_ref);
            
            for (int j = 0; j < m; j++) {
                float diff = std::abs(custom_result[j] - ref_result[j]);
                results.max_error = std::max(results.max_error, diff);
            }
        }
        
        // Cleanup
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
    }
    
    // Matrix multiplication benchmark
    std::cout << "Benchmarking matrix multiplication...\n";
    std::vector<KernelResults> matrix_results(kernels.size());
    
    for (int trial = 0; trial < num_trials; trial++) {
        // Generate random low-rank matrix
        auto [d_AB, d_A, d_B] = generateRandomLowRankMatrix(handle, gen, m, n, r);
        
        // Generate random matrix X
        CHECK_CURAND(curandGenerateNormal(gen, d_X, n * batch_size, 0.0f, 1.0f));
        
        // Compute reference result
        auto ref_kernel_data = setup_ABX(d_A, d_B, m, n, r);
        cublas_ABX(ref_kernel_data, d_X, d_output_matrix_ref, m, n, r, batch_size);
        
        // Test each implementation
        for (size_t i = 0; i < kernels.size(); i++) {
            auto& kernel = kernels[i];
            auto& results = matrix_results[i];
            
            // Time the kernel
            auto kernel_data = kernel.setup(d_A, d_B, m, n, r);
            auto kernel_time = benchmark_kernel([&]() {
                kernel.qrX_kernel(kernel_data, d_X, d_output_matrix, m, n, r, batch_size);
            });
            results.total_time += kernel_time;
            
            // Compare with reference
            thrust::device_vector<float> custom_result(m * batch_size);
            thrust::device_vector<float> ref_result(m * batch_size);
            CHECK_CUDA(cudaMemcpy(custom_result.data().get(), d_output_matrix, m * batch_size * sizeof(float), 
                      cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(ref_result.data().get(), d_output_matrix_ref, m * batch_size * sizeof(float), 
                      cudaMemcpyDeviceToDevice));
            
            for (int j = 0; j < m * batch_size; j++) {
                float diff = std::abs(custom_result[j] - ref_result[j]);
                results.max_error = std::max(results.max_error, diff);
            }
        }
        
        // Cleanup
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
    }
    
    // Print results
    std::cout << "\nVector Operation Results:\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::setw(30) << "Implementation" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Max Error" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (size_t i = 0; i < kernels.size(); i++) {
        std::cout << std::setw(30) << kernels[i].name 
                  << std::setw(15) << (vector_results[i].total_time / num_trials) / 1000
                  << std::setw(15) << vector_results[i].max_error << "\n";
    }
    
    std::cout << "\nMatrix Operation Results:\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::setw(30) << "Implementation" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Max Error" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (size_t i = 0; i < kernels.size(); i++) {
        std::cout << std::setw(30) << kernels[i].name 
                  << std::setw(15) << (matrix_results[i].total_time / num_trials) / 1000
                  << std::setw(15) << matrix_results[i].max_error << "\n";
    }
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_output_matrix));
    CHECK_CUDA(cudaFree(d_output_ref));
    CHECK_CUDA(cudaFree(d_output_matrix_ref));
    CHECK_CUDA(cudaFree(d_temp_vector));
    CHECK_CUDA(cudaFree(d_temp_matrix));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CURAND(curandDestroyGenerator(gen));
    CHECK_CUSOLVER(cusolverDnDestroy(solver_handle));

    return 0;
}