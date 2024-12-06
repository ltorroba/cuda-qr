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

struct QtKernel {
    std::string name;
    std::function<void*(int, int, const float*, float*)> preamble;  // Returns workspace pointer
    std::function<void(int, int, const float*, float*, void*)> kernel;
    std::function<void(int, int, const float*, float*, void*)> postamble;  // Can modify out if needed

    // Constructor for backward compatibility
    QtKernel(const std::string& n,
             std::function<void(int, int, const float*, float*)> k)
        : name(n)
        , preamble([](int, int, const float*, float*) -> void* { return nullptr; })
        , kernel([k](int size_in, int diag_iter, const float* tau, float* out, void*) {
            k(size_in, diag_iter, tau, out);
          })
        , postamble([](int, int, const float*, float*, void*) {}) {}

    // Constructor for full functionality
    QtKernel(const std::string& n,
             std::function<void*(int, int, const float*, float*)> pre,
             std::function<void(int, int, const float*, float*, void*)> k,
             std::function<void(int, int, const float*, float*, void*)> post)
        : name(n)
        , preamble(pre)
        , kernel(k)
        , postamble(post) {}
};

int main(int argc, char **argv) {
    bool verbose = false;
    bool memory_usage = false;
    bool single_row = false;
    int warmup_trials = 100;
    int num_trials = 100;
    int size_in = 1024;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--verbose") {
            verbose = true;
        } else if (std::string(argv[i]) == "--memory-usage") {
            memory_usage = true;
        } else if (std::string(argv[i]) == "--size") {
            size_in = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--trials") {
            num_trials = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--warmup") {
            warmup_trials = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--single-row") {
            single_row = true;
        }
    }

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
        QtKernel("cuSOLVER (Fast)",
            // Preamble
            [](int size_in, int diag_iter, const float* tau, float* matrix_out) -> void* {
                return reference_applyQt_fast_preamble(size_in, diag_iter, tau, matrix_out);
            },
            // Kernel
            [](int size_in, int diag_iter, const float* tau, float* matrix_out, void* workspace) {
                reference_applyQt_fast(size_in, diag_iter, tau, matrix_out, workspace);
            },
            // Postamble
            [](int size_in, int diag_iter, const float* tau, float* matrix_out, void* workspace) {
                reference_applyQt_fast_postamble(size_in, diag_iter, tau, matrix_out, workspace);
            }
        )
    };

    // Results structure
    struct KernelResults {
        float total_time = 0.0f;
        float max_error = 0.0f;
    };
    std::vector<KernelResults> results(kernels.size());

    int diag_iters = single_row ? 1 : size_in/tilesize - 1;

    // Benchmark loop:
    // - Initialize matrix with random data
    // - Run QR individually for each diagonal block to populate the diagonal tiles and the tau vector
    // - For each row:
    //   - Reference: Apply the Q' matrix to the tiles to the right of the diagonal and store the results
    //   - Custom: Call the kernel for this diagonal iter
    //   - Compare the results
    for (int trial = 0; trial < num_trials + warmup_trials; trial++) {
        // Print memory usage at start of each trial
        if (memory_usage && trial >= warmup_trials) {
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
        for (int diag_iter = 0; diag_iter < diag_iters; diag_iter++) {
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
        for (int diag_iter = 0; diag_iter < diag_iters; diag_iter++) {
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
                void* workspace = kernel.preamble(size_in, diag_iter, d_tau, d_matrix_out);
                float local_time = benchmark_kernel([&]() {
                    kernel.kernel(size_in, diag_iter, d_tau, d_matrix_out, workspace);
                });
                kernel.postamble(size_in, diag_iter, d_tau, d_matrix_out, workspace);
                if (trial >= warmup_trials)
                    result.total_time += local_time;

                // Copy results to host for comparison
                std::vector<float> host_custom(size_in * size_in);
                CHECK_CUDA(cudaMemcpy(host_custom.data(), d_matrix_out,
                                    size_in * size_in * sizeof(float),
                                    cudaMemcpyDeviceToHost));

                // Compare only the relevant tiles. We include the diagonal tiles since these should
                // not be modified by the kernel
                if (trial >= warmup_trials) {
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

        // Print a message after the warmup phase is complete
        if (trial == warmup_trials - 1) {
            std::cout << "Warmup phase completed. Starting timed trials...\n";
        }

        // Print a progress message every 100 trials
        if (trial > warmup_trials && (trial - warmup_trials) % 100 == 0) {
            std::cout << "Trial " << trial << " completed\n";
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