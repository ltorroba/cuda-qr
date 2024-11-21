/**************************************/
/* AUTO-GENERATED FILE -- DO NOT EDIT */
/**************************************/

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

// CUTLASS includes.
#include <cutlass/gemm/device/gemm.h>

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES));
}

__device__ __forceinline__ void async_memcpy_waitall() {
    asm volatile("cp.async.wait_all;\n" ::);
}

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)
//
// void matmul_cpu_naive(
//     int32_t size_i,
//     int32_t size_j,
//     int32_t size_k,
//     float const *a,
//     float const *b,
//     float *c) {
//     for (int32_t i = 0; i < size_i; ++i) {
//         for (int32_t j = 0; j < size_j; ++j) {
//             float sum = 0.0;
//             for (int32_t k = 0; k < size_k; ++k) {
//                 sum += a[i * size_k + k] * b[k * size_j + j];
//             }
//             c[i * size_j + j] = sum;
//         }
//     }
// }

/***********************/
/* BEGIN SOLUTION CODE */
/***********************/

#define HAS_LAB_4_BASELINE_IMPL

namespace matmul_l1_reg {

constexpr int32_t size_tile_i = 128;
constexpr int32_t size_tile_k = 32;
constexpr int32_t size_tile_j = 128;

constexpr int32_t n_thread_i = 16;
constexpr int32_t n_thread_j = 16;
constexpr int32_t n_thread = n_thread_i * n_thread_j;

static_assert(size_tile_i % n_thread_i == 0);
static_assert(size_tile_j % n_thread_j == 0);

constexpr int32_t size_micro_i = size_tile_i / n_thread_i;
constexpr int32_t size_micro_j = size_tile_j / n_thread_j;
constexpr int32_t size_micro_k = 4;

struct Shmem {
    float a_local[size_tile_i][size_tile_k];
    float b_local[size_tile_k][size_tile_j];
};

__global__ __launch_bounds__(n_thread) void matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *__restrict__ a,
    float const *__restrict__ b,
    float *__restrict__ c) {
    extern __shared__ Shmem shmem[];

    int32_t tid = threadIdx.y * n_thread_j + threadIdx.x;

    int32_t local_base_i = threadIdx.y * size_micro_i;
    int32_t local_base_j = threadIdx.x * size_micro_j;

    int32_t block_global_base_i = blockIdx.y * size_tile_i;
    int32_t block_global_base_j = blockIdx.x * size_tile_j;

    float c_micro[size_micro_i][size_micro_j] = {0.0f};

    for (int32_t global_base_k = 0; global_base_k < size_k;
         global_base_k += size_tile_k) {
        if (global_base_k > 0) {
            __syncthreads();
        }

        // copy 'a' tile to shmem
        for (int32_t local_linear = tid; local_linear < size_tile_i * size_tile_k;
             local_linear += n_thread) {
            int32_t local_i = local_linear / size_tile_k;
            int32_t local_k = local_linear % size_tile_k;
            int32_t global_i = block_global_base_i + local_i;
            int32_t global_k = global_base_k + local_k;
            shmem->a_local[local_i][local_k] = (global_i < size_i && global_k < size_k)
                ? a[global_i * size_k + global_k]
                : 0.0f;
        }
        // copy 'b' tile to shmem
        for (int32_t local_linear = tid; local_linear < size_tile_k * size_tile_j;
             local_linear += n_thread) {
            int32_t local_k = local_linear / size_tile_j;
            int32_t local_j = local_linear % size_tile_j;
            int32_t global_k = global_base_k + local_k;
            int32_t global_j = block_global_base_j + local_j;
            shmem->b_local[local_k][local_j] = (global_k < size_k && global_j < size_j)
                ? b[global_k * size_j + global_j]
                : 0.0f;
        }

        __syncthreads();

        for (int32_t local_base_k = 0; local_base_k < size_tile_k;
             local_base_k += size_micro_k) {
            float a_micro[size_micro_i][size_micro_k];
            float b_micro[size_micro_k][size_micro_j];

            // copy 'a' microtile to register

#pragma unroll
            for (int32_t micro_i = 0; micro_i < size_micro_i; micro_i++) {
                int32_t local_i = local_base_i + micro_i;
#pragma unroll
                for (int32_t micro_k = 0; micro_k < size_micro_k; micro_k++) {
                    int32_t local_k = local_base_k + micro_k;
                    a_micro[micro_i][micro_k] = shmem->a_local[local_i][local_k];
                }
            }

            // copy 'b' microtile to register

#pragma unroll
            for (int32_t micro_k = 0; micro_k < size_micro_k; micro_k++) {
                int32_t local_k = local_base_k + micro_k;
#pragma unroll
                for (int32_t micro_j = 0; micro_j < size_micro_j; micro_j++) {
                    int32_t local_j = local_base_j + micro_j;
                    b_micro[micro_k][micro_j] = shmem->b_local[local_k][local_j];
                }
            }

            // compute

#pragma unroll
            for (int32_t micro_i = 0; micro_i < size_micro_i; micro_i++) {
#pragma unroll
                for (int32_t micro_j = 0; micro_j < size_micro_j; micro_j++) {
#pragma unroll
                    for (int32_t micro_k = 0; micro_k < size_micro_k; micro_k++) {
                        c_micro[micro_i][micro_j] +=
                            a_micro[micro_i][micro_k] * b_micro[micro_k][micro_j];
                    }
                }
            }
        }
    }

    // copy 'c' microtile to global memory

#pragma unroll
    for (int32_t micro_i = 0; micro_i < size_micro_i; micro_i++) {
        int32_t global_i = block_global_base_i + local_base_i + micro_i;
#pragma unroll
        for (int32_t micro_j = 0; micro_j < size_micro_j; micro_j++) {
            int32_t global_j = block_global_base_j + local_base_j + micro_j;
            if (global_i < size_i && global_j < size_j) {
                c[global_i * size_j + global_j] = c_micro[micro_i][micro_j];
            }
        }
    }
}

void launch_matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_l1_reg,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(Shmem)));

    dim3 block_size(n_thread_j, n_thread_i);
    dim3 grid_size(
        (size_j + size_tile_j - 1) / size_tile_j,
        (size_i + size_tile_i - 1) / size_tile_i);
    matmul_l1_reg<<<grid_size, block_size, sizeof(Shmem)>>>(
        size_i,
        size_j,
        size_k,
        a,
        b,
        c);
    CUDA_CHECK(cudaGetLastError());
}

}; // namespace matmul_l1_reg

namespace matmul_improved {

constexpr int32_t size_tile_i = 128;
constexpr int32_t size_tile_k = 32;
constexpr int32_t size_tile_j = 128;

constexpr int32_t n_thread_i = 16;
constexpr int32_t n_thread_j = 16;
constexpr int32_t n_thread = n_thread_i * n_thread_j;

static_assert(size_tile_i % n_thread_i == 0);
static_assert(size_tile_j % n_thread_j == 0);

constexpr int32_t size_micro_i = size_tile_i / n_thread_i;
constexpr int32_t size_micro_j = size_tile_j / n_thread_j;
constexpr int32_t size_micro_k = 4;

struct Shmem {
    float a_local[size_tile_i][size_tile_k];
    float b_local[size_tile_k][size_tile_j];
};

__global__ __launch_bounds__(n_thread) void matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *__restrict__ a,
    float const *__restrict__ b,
    float *__restrict__ c) {
    extern __shared__ Shmem shmem[];

    int32_t tid = threadIdx.y * n_thread_j + threadIdx.x;

    int32_t local_base_i = threadIdx.y * size_micro_i;
    int32_t local_base_j = threadIdx.x * size_micro_j;

    int32_t block_global_base_i = blockIdx.y * size_tile_i;
    int32_t block_global_base_j = blockIdx.x * size_tile_j;

    float c_micro[size_micro_i][size_micro_j] = {0.0f};

    for (int32_t global_base_k = 0; global_base_k < size_k;
         global_base_k += size_tile_k) {
        if (global_base_k > 0) {
            __syncthreads();
        }

        // copy 'a' tile to shmem
        for (int32_t local_linear = tid; local_linear < size_tile_i * size_tile_k;
             local_linear += n_thread) {
            int32_t local_i = local_linear / size_tile_k;
            int32_t local_k = local_linear % size_tile_k;
            int32_t global_i = block_global_base_i + local_i;
            int32_t global_k = global_base_k + local_k;
            shmem->a_local[local_i][local_k] = (global_i < size_i && global_k < size_k)
                ? a[global_i * size_k + global_k]
                : 0.0f;
        }
        // copy 'b' tile to shmem
        for (int32_t local_linear = tid; local_linear < size_tile_k * size_tile_j;
             local_linear += n_thread) {
            int32_t local_k = local_linear / size_tile_j;
            int32_t local_j = local_linear % size_tile_j;
            int32_t global_k = global_base_k + local_k;
            int32_t global_j = block_global_base_j + local_j;
            shmem->b_local[local_k][local_j] = (global_k < size_k && global_j < size_j)
                ? b[global_k * size_j + global_j]
                : 0.0f;
        }

        __syncthreads();

        for (int32_t local_base_k = 0; local_base_k < size_tile_k;
             local_base_k += size_micro_k) {
            float a_micro[size_micro_i][size_micro_k];
            float b_micro[size_micro_k][size_micro_j];

            // copy 'a' microtile to register

#pragma unroll
            for (int32_t micro_i = 0; micro_i < size_micro_i; micro_i++) {
                int32_t local_i = local_base_i + micro_i;
#pragma unroll
                for (int32_t micro_k = 0; micro_k < size_micro_k; micro_k++) {
                    int32_t local_k = local_base_k + micro_k;
                    a_micro[micro_i][micro_k] = shmem->a_local[local_i][local_k];
                }
            }

            // copy 'b' microtile to register

#pragma unroll
            for (int32_t micro_k = 0; micro_k < size_micro_k; micro_k++) {
                int32_t local_k = local_base_k + micro_k;
#pragma unroll
                for (int32_t micro_j = 0; micro_j < size_micro_j; micro_j++) {
                    int32_t local_j = local_base_j + micro_j;
                    b_micro[micro_k][micro_j] = shmem->b_local[local_k][local_j];
                }
            }

            // compute

#pragma unroll
            for (int32_t micro_i = 0; micro_i < size_micro_i; micro_i++) {
#pragma unroll
                for (int32_t micro_j = 0; micro_j < size_micro_j; micro_j++) {
#pragma unroll
                    for (int32_t micro_k = 0; micro_k < size_micro_k; micro_k++) {
                        c_micro[micro_i][micro_j] +=
                            a_micro[micro_i][micro_k] * b_micro[micro_k][micro_j];
                    }
                }
            }
        }
    }

    // copy 'c' microtile to global memory

#pragma unroll
    for (int32_t micro_i = 0; micro_i < size_micro_i; micro_i++) {
        int32_t global_i = block_global_base_i + local_base_i + micro_i;
#pragma unroll
        for (int32_t micro_j = 0; micro_j < size_micro_j; micro_j++) {
            int32_t global_j = block_global_base_j + local_base_j + micro_j;
            if (global_i < size_i && global_j < size_j) {
                c[global_i * size_j + global_j] = c_micro[micro_i][micro_j];
            }
        }
    }
}

void launch_matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_improved,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(Shmem)));

    dim3 block_size(n_thread_j, n_thread_i);
    dim3 grid_size(
        (size_j + size_tile_j - 1) / size_tile_j,
        (size_i + size_tile_i - 1) / size_tile_i);
    matmul_improved<<<grid_size, block_size, sizeof(Shmem)>>>(
        size_i,
        size_j,
        size_k,
        a,
        b,
        c);
    CUDA_CHECK(cudaGetLastError());
}

}; // namespace matmul_improved

namespace matmul_improved_reduce {

constexpr int32_t size_tile_i = 128;
constexpr int32_t size_tile_k = 32;
constexpr int32_t size_tile_j = 128;

constexpr int32_t n_thread_i = 16;
constexpr int32_t n_thread_j = 16;
constexpr int32_t n_thread = n_thread_i * n_thread_j;

static_assert(size_tile_i % n_thread_i == 0);
static_assert(size_tile_j % n_thread_j == 0);

constexpr int32_t size_micro_i = size_tile_i / n_thread_i;
constexpr int32_t size_micro_j = size_tile_j / n_thread_j;
constexpr int32_t size_micro_k = 4;

struct Shmem {
    float a_local[size_tile_i][size_tile_k];
    float b_local[size_tile_k][size_tile_j];
};

int32_t ceil_div(int32_t a, int32_t b) { return (a + b - 1) / b; }

constexpr int32_t num_sms = 48;

int32_t get_k_slices(int32_t size_i, int32_t size_j, int32_t k) {
    int32_t num_ij_tiles = ceil_div(size_i, size_tile_i) * ceil_div(size_j, size_tile_j);
    return ceil_div(num_sms, num_ij_tiles);
}

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    int32_t num_k_slices = get_k_slices(size_i, size_j, size_k);
    return size_i * size_j * num_k_slices * sizeof(float);
}

__global__ __launch_bounds__(n_thread) void matmul_improved_reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    int32_t size_slice_k,
    float const *__restrict__ a,
    float const *__restrict__ b,
    float *workspace) {
    extern __shared__ Shmem shmem[];

    int32_t tid = threadIdx.y * n_thread_j + threadIdx.x;

    int32_t local_base_i = threadIdx.y * size_micro_i;
    int32_t local_base_j = threadIdx.x * size_micro_j;

    int32_t block_global_base_k = blockIdx.z * size_slice_k;
    int32_t block_global_base_i = blockIdx.y * size_tile_i;
    int32_t block_global_base_j = blockIdx.x * size_tile_j;

    float c_micro[size_micro_i][size_micro_j] = {0.0f};

    for (int32_t global_base_k = block_global_base_k;
         global_base_k < block_global_base_k + size_slice_k;
         global_base_k += size_tile_k) {
        if (global_base_k > block_global_base_k) {
            __syncthreads();
        }

        // copy 'a' tile to shmem
        for (int32_t local_linear = tid; local_linear < size_tile_i * size_tile_k;
             local_linear += n_thread) {
            int32_t local_i = local_linear / size_tile_k;
            int32_t local_k = local_linear % size_tile_k;
            int32_t global_i = block_global_base_i + local_i;
            int32_t global_k = global_base_k + local_k;
            shmem->a_local[local_i][local_k] =
                (global_i < size_i && global_k < size_k &&
                 global_k < block_global_base_k + size_slice_k)
                ? a[global_i * size_k + global_k]
                : 0.0f;
        }
        // copy 'b' tile to shmem
        for (int32_t local_linear = tid; local_linear < size_tile_k * size_tile_j;
             local_linear += n_thread) {
            int32_t local_k = local_linear / size_tile_j;
            int32_t local_j = local_linear % size_tile_j;
            int32_t global_k = global_base_k + local_k;
            int32_t global_j = block_global_base_j + local_j;
            shmem->b_local[local_k][local_j] =
                (global_k < size_k && global_k < block_global_base_k + size_slice_k &&
                 global_j < size_j)
                ? b[global_k * size_j + global_j]
                : 0.0f;
        }

        __syncthreads();

        for (int32_t local_base_k = 0; local_base_k < size_tile_k;
             local_base_k += size_micro_k) {
            float a_micro[size_micro_i][size_micro_k];
            float b_micro[size_micro_k][size_micro_j];

            // copy 'a' microtile to register

#pragma unroll
            for (int32_t micro_i = 0; micro_i < size_micro_i; micro_i++) {
                int32_t local_i = local_base_i + micro_i;
#pragma unroll
                for (int32_t micro_k = 0; micro_k < size_micro_k; micro_k++) {
                    int32_t local_k = local_base_k + micro_k;
                    a_micro[micro_i][micro_k] = shmem->a_local[local_i][local_k];
                }
            }

            // copy 'b' microtile to register

#pragma unroll
            for (int32_t micro_k = 0; micro_k < size_micro_k; micro_k++) {
                int32_t local_k = local_base_k + micro_k;
#pragma unroll
                for (int32_t micro_j = 0; micro_j < size_micro_j; micro_j++) {
                    int32_t local_j = local_base_j + micro_j;
                    b_micro[micro_k][micro_j] = shmem->b_local[local_k][local_j];
                }
            }

            // compute

#pragma unroll
            for (int32_t micro_i = 0; micro_i < size_micro_i; micro_i++) {
#pragma unroll
                for (int32_t micro_j = 0; micro_j < size_micro_j; micro_j++) {
#pragma unroll
                    for (int32_t micro_k = 0; micro_k < size_micro_k; micro_k++) {
                        c_micro[micro_i][micro_j] +=
                            a_micro[micro_i][micro_k] * b_micro[micro_k][micro_j];
                    }
                }
            }
        }
    }

    // copy 'c' microtile to global memory

    float *c_partial = workspace + blockIdx.z * size_i * size_j;

#pragma unroll
    for (int32_t micro_i = 0; micro_i < size_micro_i; micro_i++) {
        int32_t global_i = block_global_base_i + local_base_i + micro_i;
#pragma unroll
        for (int32_t micro_j = 0; micro_j < size_micro_j; micro_j++) {
            int32_t global_j = block_global_base_j + local_base_j + micro_j;
            if (global_i < size_i && global_j < size_j) {
                c_partial[global_i * size_j + global_j] = c_micro[micro_i][micro_j];
            }
        }
    }
}

__global__ void matmul_improved_reduce_final(
    int32_t size,
    int32_t num_k_slices,
    float const *workspace,
    float *c) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    float sum = 0.0f;
    for (int32_t slice = 0; slice < num_k_slices; slice++) {
        sum += workspace[slice * size + idx];
    }

    c[idx] = sum;
}

void launch_matmul_improved_reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c,
    void *workspace) {
    int32_t num_k_slices = get_k_slices(size_i, size_j, size_k);
    int32_t size_slice_k = ceil_div(size_k, num_k_slices);

    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_improved_reduce,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(Shmem)));

    dim3 block_size(n_thread_j, n_thread_i);
    dim3 grid_size(
        (size_j + size_tile_j - 1) / size_tile_j,
        (size_i + size_tile_i - 1) / size_tile_i,
        num_k_slices);
    matmul_improved_reduce<<<grid_size, block_size, sizeof(Shmem)>>>(
        size_i,
        size_j,
        size_k,
        size_slice_k,
        a,
        b,
        reinterpret_cast<float *>(workspace));
    CUDA_CHECK(cudaGetLastError());
    matmul_improved_reduce_final<<<ceil_div(size_i * size_j, 1024), 1024>>>(
        size_i * size_j,
        num_k_slices,
        reinterpret_cast<float *>(workspace),
        c);
    CUDA_CHECK(cudaGetLastError());
}

}; // namespace matmul_improved_reduce

namespace matmul_cutlass {

void launch_matmul_cutlass(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c,
    void *workspace) {
    // Define data types for the computation.
    using ElementInputA = float;
    using ElementInputB = float;
    using ElementOutput = float;
    using ElementAccumulator = float;
    using ElementCompute = float;

    // Define the layouts of the matrices (row-major in this case).
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    // Define the GEMM operation.
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator>;

    // Create GEMM arguments.
    ElementCompute alpha = 1.0f;
    ElementCompute beta = 0.0f;
    typename Gemm::Arguments args(
        {size_i, size_j, size_k},
        {a, size_k},     // Tensor A (device pointer and leading dimension)
        {b, size_j},     // Tensor B (device pointer and leading dimension)
        {c, size_j},     // Tensor C (device pointer and leading dimension)
        {c, size_j},     // Tensor D (output tensor)
        {alpha, beta}      // Scalars used in the epilogue
    );
    Gemm gemm_op;

    // Check and init GEMM.
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cout << "Unsupported operation\n";
        return;
    }

    status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        std::cout << "Failed to init GEMM\n";
        return;
    }

    // Launch GEMM.
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        std::cout << "Failed to run GEMM\n";
        return;
    }
}

}; // namespace matmul_cutlass

/***********************/
/* END SOLUTION CODE   */
/***********************/


////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

template <typename Reset, typename F>
double
benchmark_ms(double target_time_ms, int32_t num_iters_inner, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
};

struct TestData {
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> b;
    std::map<std::tuple<int32_t, int32_t, int32_t>, std::vector<float>> c;
};

TestData read_test_data(
    std::string const &test_data_dir,
    std::vector<BenchmarkConfig> const &configs) {
    auto data = TestData{};
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_";

        if (data.a.find({size_i, size_k}) == data.a.end()) {
            data.a[{size_i, size_k}] = read_data(
                path_prefix + "a_" + std::to_string(size_i) + "x" +
                    std::to_string(size_k) + ".bin",
                size_i * size_k);
        }

        if (data.b.find({size_k, size_j}) == data.b.end()) {
            data.b[{size_k, size_j}] = read_data(
                path_prefix + "b_" + std::to_string(size_k) + "x" +
                    std::to_string(size_j) + ".bin",
                size_k * size_j);
        }

        if (data.c.find({size_i, size_j, size_k}) == data.c.end()) {
            data.c[{size_i, size_j, size_k}] = read_data(
                path_prefix + "c_" + std::to_string(size_i) + "x" +
                    std::to_string(size_j) + "x" + std::to_string(size_k) + ".bin",
                size_i * size_j);
        }
    }
    return data;
}

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t, int32_t, int32_t>, double> elapsed_ms;
};

enum class Phase {
    WARMUP,
    BENCHMARK,
};

template <typename Impl>
void run_config(
    Phase phase,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size_i = config.size_i;
    auto size_j = config.size_j;
    auto size_k = config.size_k;

    auto const &a = data.a.at({size_i, size_k});
    auto const &b = data.b.at({size_k, size_j});
    auto const &c = data.c.at({size_i, size_j, size_k});

    float *a_gpu;
    float *b_gpu;
    float *c_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size_i * size_k * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        b_gpu,
        b.data(),
        size_k * size_j * sizeof(float),
        cudaMemcpyHostToDevice));

    size_t workspace_size = Impl::get_workspace_size(size_i, size_j, size_k);
    void *workspace_gpu = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
    }

    if (phase == Phase::BENCHMARK) {
        printf("  %6d  %6d  %6d", size_i, size_j, size_k);
    } else {
        printf("  warmup %6d  %6d  %6d", size_i, size_j, size_k);
    }

    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);

    std::vector<float> c_out_host(size_i * size_j);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        c_gpu,
        size_i * size_j * sizeof(float),
        cudaMemcpyDeviceToHost));

    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < size_i; ++i) {
        for (int32_t j = 0; j < size_j; ++j) {
            float diff = c_out_host[i * size_j + j] - c[i * size_j + j];
            mse += diff * diff;
            ref_mean_square += c[i * size_j + j] * c[i * size_j + j];
        }
    }
    mse /= size_i * size_j;
    ref_mean_square /= size_i * size_j;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);

    if (phase == Phase::BENCHMARK) {
        printf("  %8.02e", rel_rmse);
    }

    if (rel_rmse > 1e-5) {
        if (phase == Phase::BENCHMARK) {
            printf("  %9s  %7s", "-", "-");
        }
    } else {
        double target_time_ms = 200.0;
        double elapsed_ms = benchmark_ms(
            target_time_ms,
            4,
            [&]() {
                if (workspace_size > 0) {
                    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                }
            },
            [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);
            });

        if (phase == Phase::BENCHMARK) {
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("  %9.02f  %7.02f", elapsed_ms, tflop / (elapsed_ms * 1e-3));

            results.elapsed_ms[{size_i, size_j, size_k}] = elapsed_ms;
        }
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
}

template <typename Impl>
BenchmarkResults run_all_configs(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{Impl::name};
    if (phase == Phase::WARMUP) {
        printf("warmup %s:\n\n", Impl::name);
    } else {
        printf("%s:\n\n", Impl::name);
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "size_i",
            "size_j",
            "size_k",
            "RRMSE",
            "time (ms)",
            "TFLOP/s");
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "------",
            "------",
            "------",
            "--------",
            "---------",
            "-------");
    }
    for (auto const &config : configs) {
        run_config<Impl>(phase, data, config, results);
    }
    printf("\n");
    return results;
}

#ifdef HAS_LAB_4_BASELINE_IMPL

struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
    }
};

#endif

struct MatmulImproved {
    constexpr static char const *name = "matmul_improved";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved::launch_matmul_improved(size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulImprovedReduce {
    constexpr static char const *name = "matmul_improved_reduce";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_improved_reduce::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved_reduce::launch_matmul_improved_reduce(
            size_i,
            size_j,
            size_k,
            a,
            b,
            c,
            workspace);
    }
};

struct MatmulCUTLASS {
    constexpr static char const *name = "matmul_cutlass";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_cutlass::launch_matmul_cutlass(
            size_i,
            size_j,
            size_k,
            a,
            b,
            c,
            workspace);
    }
};

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
#ifdef HAS_LAB_4_BASELINE_IMPL
    results.push_back(run_all_configs<MatmulL1Reg>(phase, data, configs));
#endif
    results.push_back(run_all_configs<MatmulImproved>(phase, data, configs));
    results.push_back(run_all_configs<MatmulImprovedReduce>(phase, data, configs));
    results.push_back(run_all_configs<MatmulCUTLASS>(phase, data, configs));
    return results;
}

void write_json_results(
    std::string const &path,
    std::vector<BenchmarkResults> const &results) {
    auto file = std::ofstream(path);
    file << "{\n";
    for (int32_t i = 0; i < results.size(); ++i) {
        auto const &result = results.at(i);
        file << "  \"" << result.name << "\": [\n";
        int32_t j = 0;
        for (auto const &[config, elapsed_ms] : result.elapsed_ms) {
            auto [size_i, size_j, size_k] = config;
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            double tflop_per_sec = tflop / (elapsed_ms * 1e-3);
            file << "    {\n";
            file << "      \"size_i\": " << size_i << ",\n";
            file << "      \"size_j\": " << size_j << ",\n";
            file << "      \"size_k\": " << size_k << ",\n";
            file << "      \"elapsed_ms\": " << elapsed_ms << ",\n";
            file << "      \"tflop_per_sec\": " << tflop_per_sec << "\n";
            file << "    }";
            if (j + 1 < result.elapsed_ms.size()) {
                file << ",";
            }
            file << "\n";
            ++j;
        }
        file << "  ]";
        if (i + 1 < results.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "}\n";
}

int main(int argc, char **argv) {
    std::string test_data_dir = ".";
    if (char *c_str_test_data_dir = std::getenv("MATMUL_TEST_DATA_DIR_2")) {
        test_data_dir = c_str_test_data_dir;
    }

    auto configs = std::vector<BenchmarkConfig>{
        {3072, 3072, 3072},
        {512, 3072, 3072},
        {256, 3072, 3072},
        {128, 3072, 3072},
        {64, 3072, 3072},
        {32, 3072, 3072},
        {16, 3072, 3072},
        {1, 3072, 3072},
        {256, 256, 256},
        {256, 256, 1024},
        {256, 256, 8192},
        {128, 128, 32768},
    };
    auto data = read_test_data(test_data_dir, configs);
    run_all_impls(Phase::WARMUP, data, configs);
    auto results = run_all_impls(Phase::BENCHMARK, data, configs);

    for (int32_t j = 1; j < results.size(); ++j) {
        for (int32_t i = j; i > 0;) {
            --i;
            auto const &first = results.at(i);
            auto const &second = results.at(j);
            printf("\nspeedups %s -> %s:\n\n", first.name, second.name);
            printf("  %-6s  %-6s  %-6s  %-7s\n", "size_i", "size_j", "size_k", "speedup");
            printf("  %-6s  %-6s  %-6s  %-7s\n", "------", "------", "------", "-------");
            for (auto const &config : configs) {
                auto size_i = config.size_i;
                auto size_j = config.size_j;
                auto size_k = config.size_k;
                printf("  %6d  %6d  %6d", size_i, size_j, size_k);
                auto it_first = first.elapsed_ms.find({size_i, size_j, size_k});
                auto it_second = second.elapsed_ms.find({size_i, size_j, size_k});
                if (it_first != first.elapsed_ms.end() &&
                    it_second != second.elapsed_ms.end()) {
                    printf("  %6.02fx", it_first->second / it_second->second);
                } else {
                    printf("  %7s", "-");
                }
                printf("\n");
            }
        }
    }

    write_json_results("out/results.json", results);

    return 0;
}
