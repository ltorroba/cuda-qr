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

struct QtKernel {
    std::string name;
    std::function<void(int, int, const float*, float*)> kernel;
    
    QtKernel(const std::string& n,
             std::function<void(int, int, const float*, float*)> k)
        : name(n), kernel(k) {}
};

int main(int argc, char **argv) {
    const int num_trials = 1;
    const int size_in = 256;  // matrix size
    constexpr int tilesize = 32;  // tile size
    constexpr int numthreads = 32;  // compile-time constant
    
    // Initialize CUDA resources
    cusolverDnHandle_t solver_handle;
    CHECK_CUSOLVER(cusolverDnCreate(&solver_handle));

    // Allocate device memory
    float *d_matrix, *d_matrix_ref, *d_tau;
    CHECK_CUDA(cudaMalloc(&d_matrix, size_in * size_in * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_matrix_ref, size_in * size_in * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tau, size_in * sizeof(float)));

    // Define kernel implementations
    std::vector<QtKernel> kernels = {
        QtKernel("Original Implementation",
            [](int size_in, int diag_iter, const float* tau, float* matrix) {
                dim3 block(tilesize);
                dim3 grid((size_in - diag_iter * tilesize) / tilesize - 1);
                base_applyQt_singletile<tilesize, numthreads><<<grid, block>>>(size_in, diag_iter, tau, matrix);
                CHECK_CUDA(cudaDeviceSynchronize());
            }
        ),
        QtKernel("Reference Implementation",
            reference_applyQt
        )
    };

    // Results structure
    struct KernelResults {
        float total_time = 0.0f;
        float max_error = 0.0f;
    };
    std::vector<KernelResults> results(kernels.size());

    // Benchmark loop
    for (int trial = 0; trial < num_trials; trial++) {
        // Initialize matrix with random data
        curandGenerator_t gen;
        CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL + trial));
        CHECK_CURAND(curandGenerateNormal(gen, d_matrix, size_in * size_in, 0.0f, 1.0f));
        CHECK_CURAND(curandGenerateNormal(gen, d_tau, size_in, 0.0f, 1.0f));
        CHECK_CURAND(curandDestroyGenerator(gen));

        // For each diagonal block
        for (int diag_iter = 0; diag_iter < size_in/tilesize - 1; diag_iter++) {
            // Copy fresh matrix for each implementation
            CHECK_CUDA(cudaMemcpy(d_matrix_ref, d_matrix, 
                                size_in * size_in * sizeof(float), 
                                cudaMemcpyDeviceToDevice));
            
            // Test each implementation
            for (size_t i = 0; i < kernels.size(); i++) {
                auto& kernel = kernels[i];
                auto& result = results[i];
                
                // Time the kernel
                auto start = std::chrono::high_resolution_clock::now();
                kernel.kernel(size_in, diag_iter, d_tau, 
                            (i == 0) ? d_matrix : d_matrix_ref);
                auto end = std::chrono::high_resolution_clock::now();
                result.total_time += std::chrono::duration<float, std::micro>(end - start).count();
                
                // Compare results after first implementation
                if (i > 0) {
                    thrust::device_ptr<float> custom_result = thrust::device_pointer_cast(d_matrix);
                    thrust::device_ptr<float> ref_result = thrust::device_pointer_cast(d_matrix_ref);
                    
                    for (int j = 0; j < size_in * size_in; j++) {
                        float diff = std::abs(custom_result[j] - ref_result[j]);
                        result.max_error = std::max(result.max_error, diff);
                        if (diff > 1e-5) {
                            int row = j % size_in;
                            int col = j / size_in;
                            std::cout << "Large difference at (" << row << "," << col << "): "
                                    << "custom=" << custom_result[j] 
                                    << " ref=" << ref_result[j] 
                                    << " diff=" << diff << "\n";
                        }
                    }
                }
            }
        }
    }
    
    // Print results
    std::cout << "\nResults:\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::setw(30) << "Implementation" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Max Error" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (size_t i = 0; i < kernels.size(); i++) {
        std::cout << std::setw(30) << kernels[i].name 
                  << std::setw(15) << results[i].total_time / 1000.0f
                  << std::setw(15) << results[i].max_error << "\n";
    }
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_matrix));
    CHECK_CUDA(cudaFree(d_matrix_ref));
    CHECK_CUDA(cudaFree(d_tau));
    CHECK_CUSOLVER(cusolverDnDestroy(solver_handle));

    return 0;
}