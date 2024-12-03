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
double benchmark_kernel(F func) {
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    func();
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

void convert_matrix_major(const float* input_d,
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

struct QtKernel {
    std::string name;
    std::function<void(int, int, const float*, float*)> kernel;
    
    QtKernel(const std::string& n,
             std::function<void(int, int, const float*, float*)> k)
        : name(n), kernel(k) {}
};

int main(int argc, char **argv) {
    bool verbose = false;
    bool memory_usage = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--verbose") {
            verbose = true;
            break;
        } else if (std::string(argv[i]) == "--memory-usage") {
            memory_usage = true;
            break;
        }
    }

    const int num_trials = 100;
    // TODO: Fix for larger matrix sizes (e.g., 96)
    const int size_in = 1024;  // matrix size
    constexpr int tilesize = 32;  // tile size
    constexpr int numthreads = 4;  // compile-time constant
    
    // Initialize CUDA resources
    cusolverDnHandle_t solver_handle;
    CHECK_CUSOLVER(cusolverDnCreate(&solver_handle));

    // Allocate device memory
    float *d_matrix_input, *d_matrix, *d_matrix_out, *d_matrix_out_ref, *d_tau;
    CHECK_CUDA(cudaMalloc(&d_matrix_input, size_in * size_in * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_matrix, size_in * size_in * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_matrix_out, size_in * size_in * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_matrix_out_ref, size_in * size_in * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tau, (size_in / tilesize) * size_in * sizeof(float)));

    // Define kernel implementations
    std::vector<QtKernel> kernels = {
        QtKernel("Original (Evelyne)",
            launch_base_applyQt_singletile_evelyne
        ),
        QtKernel("Improved (Lucas)",
            launch_base_applyQt_singletile
        ),
        QtKernel("Reference Implementation",
            // TODO: This is unfair to cuBLAS; should use efficient kernel & in col major
            // TODO: Maybe we could introduce optional preamble and postamble functions, with
            //       a shared pointer between them?
            [&](int size_in, int diag_iter, const float* tau, float* matrix_out) {
                // Allocate memory for column major result
                float* matrix_out_col_major;
                CHECK_CUDA(cudaMalloc(&matrix_out_col_major, size_in * size_in * sizeof(float)));
                convert_matrix_major(matrix_out, matrix_out_col_major, size_in, size_in);
                reference_applyQt(size_in, diag_iter, tau, matrix_out_col_major);
                convert_matrix_major(matrix_out_col_major, matrix_out, size_in, size_in, false);
                CHECK_CUDA(cudaFree(matrix_out_col_major));
            }
        )
    };

    // Results structure
    struct KernelResults {
        float total_time = 0.0f;
        float max_error = 0.0f;
    };
    std::vector<KernelResults> results(kernels.size());

    // Benchmark loop:
    // - Initialize matrix with random data
    // - Run QR individually for each diagonal block to populate the diagonal tiles and the tau vector
    // - For each row:
    //   - Reference: Apply the Q' matrix to the tiles to the right of the diagonal and store the results
    //   - Custom: Call the kernel for this diagonal iter
    //   - Compare the results
    for (int trial = 0; trial < num_trials; trial++) {
        // Print memory usage at start of each trial
        if (memory_usage) {
            size_t free_byte, total_byte;
            CHECK_CUDA(cudaMemGetInfo(&free_byte, &total_byte));
            float free_gb = free_byte / (1024.0 * 1024.0 * 1024.0);
            float total_gb = total_byte / (1024.0 * 1024.0 * 1024.0);
            float used_gb = total_gb - free_gb;
            printf("\nTrial %d - Memory Usage: Used = %.2f GB, Free = %.2f GB, Total = %.2f GB\n",
               trial, used_gb, free_gb, total_gb);
        }

        // Initialize matrix with random data
        curandGenerator_t gen;
        CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL + trial));
        CHECK_CURAND(curandGenerateNormal(gen, d_matrix_input, size_in * size_in, 0.0f, 1.0f));
        CHECK_CURAND(curandDestroyGenerator(gen));

        // Run QR individually for each diagonal block to populate the diagonal tiles and the tau vector
        for (int diag_iter = 0; diag_iter < size_in/tilesize - 1; diag_iter++) {
            // Extract the diagonal tile
            float* diag_tile;
            CHECK_CUDA(cudaMalloc(&diag_tile, tilesize * tilesize * sizeof(float)));
            
            // Copy the diagonal tile from the matrix
            for(int j = 0; j < tilesize; j++) {
                CHECK_CUDA(cudaMemcpy(diag_tile + j * tilesize,
                                    d_matrix_input + (diag_iter * tilesize + j) * size_in + diag_iter * tilesize,
                                    tilesize * sizeof(float),
                                    cudaMemcpyDeviceToDevice));
            }
            
            // Perform QR on this tile
            int* devInfo;
            CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
            
            // Query workspace size
            int lwork;
            CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(
                solver_handle,
                tilesize, tilesize,
                diag_tile,
                tilesize,
                &lwork));
                
            // Allocate workspace
            float* workspace;
            CHECK_CUDA(cudaMalloc(&workspace, lwork * sizeof(float)));
            
            // Compute QR factorization
            CHECK_CUSOLVER(cusolverDnSgeqrf(
                solver_handle,
                tilesize, tilesize,
                diag_tile,
                tilesize,
                d_tau + diag_iter * size_in,  // TODO: Clarify why this is the right format for tau
                workspace,
                lwork,
                devInfo));
                
            // Copy the result back to the matrix
            for(int j = 0; j < tilesize; j++) {
                CHECK_CUDA(cudaMemcpy(d_matrix_input + (diag_iter * tilesize + j) * size_in + diag_iter * tilesize,
                                    diag_tile + j * tilesize,
                                    tilesize * sizeof(float),
                                    cudaMemcpyDeviceToDevice));
            }
            
            // Cleanup
            CHECK_CUDA(cudaFree(workspace));
            CHECK_CUDA(cudaFree(devInfo));
            CHECK_CUDA(cudaFree(diag_tile));
        }

        // For each diagonal block
        for (int diag_iter = 0; diag_iter < size_in/tilesize - 1; diag_iter++) {
            // Copy fresh matrix for each implementation, i.e., discard previous changes
            CHECK_CUDA(cudaMemcpy(d_matrix, d_matrix_input, 
                                size_in * size_in * sizeof(float), 
                                cudaMemcpyDeviceToDevice));

            // Compute reference results for this diagonal iter
            CHECK_CUDA(cudaMemcpy(d_matrix_out_ref, d_matrix,
                                size_in * size_in * sizeof(float),
                                cudaMemcpyDeviceToDevice));
            reference_applyQt(size_in, diag_iter, d_tau, d_matrix_out_ref);

            std::vector<float> host_ref(size_in * size_in);
            CHECK_CUDA(cudaMemcpy(host_ref.data(), d_matrix_out_ref,
                                size_in * size_in * sizeof(float),
                                cudaMemcpyDeviceToHost));
            
            // Test each implementation
            for (size_t i = 0; i < kernels.size(); i++) {
                // Copy input from column major (default cuBLAS format) to row major
                convert_matrix_major(d_matrix, d_matrix_out, size_in, size_in);

                auto& kernel = kernels[i];
                auto& result = results[i];
                
                // Time the kernel
                result.total_time += benchmark_kernel([&]() {
                    kernel.kernel(size_in, diag_iter, d_tau, d_matrix_out);
                });
                
                // Copy results to host for comparison
                std::vector<float> host_custom(size_in * size_in);
                CHECK_CUDA(cudaMemcpy(host_custom.data(), d_matrix_out,
                                    size_in * size_in * sizeof(float),
                                    cudaMemcpyDeviceToHost));
                
                // Compare only the relevant tiles. We include the diagonal tiles since these should
                // not be modified by the kernel
                for (int tile = diag_iter; tile < size_in/tilesize; tile++) {
                    for (int row = diag_iter * tilesize; row < (diag_iter + 1) * tilesize; row++) {
                        for (int col = tile * tilesize; col < (tile + 1) * tilesize; col++) {
                            int col_major_idx = col * size_in + row;
                            int row_major_idx = row * size_in + col;
                            float diff = std::abs(host_custom[row_major_idx] - host_ref[col_major_idx]);
                            result.max_error = std::max(result.max_error, diff);

                            // std::cout << "tile: " << tile << " row: " << row << " col: " << col << "\n";
                            // std::cout << "host_custom[" << row_major_idx << "] = " << host_custom[row_major_idx] << "\n";
                            // std::cout << "host_ref[" << col_major_idx << "] = " << host_ref[col_major_idx] << "\n";

                            if (diff > 1e-5 && verbose) {
                                std::cout << "Large difference at tile (" 
                                        << row/tilesize << "," << col/tilesize 
                                        << ") rel_position (" << row % tilesize << "," << col % tilesize 
                                        << ") abs_position (" << row << "," << col << ")\n"
                                        << "): custom=" << host_custom[row_major_idx] 
                                        << " ref=" << host_ref[col_major_idx] 
                                        << " diff=" << diff << "\n";
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Print results
    std::cout << "\nResults (averaged over " << num_trials << " trials) for (" << size_in << "x" << size_in << "):\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::setw(30) << "Implementation" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Max Error" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (size_t i = 0; i < kernels.size(); i++) {
        std::cout << std::setw(30) << kernels[i].name 
                  << std::setw(15) << results[i].total_time / num_trials / 1000.0f
                  << std::setw(15) << results[i].max_error << "\n";
    }
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_matrix_input));
    CHECK_CUDA(cudaFree(d_matrix));
    CHECK_CUDA(cudaFree(d_matrix_out));
    CHECK_CUDA(cudaFree(d_matrix_out_ref));
    CHECK_CUDA(cudaFree(d_tau));
    CHECK_CUSOLVER(cusolverDnDestroy(solver_handle));

    return 0;
}