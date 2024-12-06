#include "reference_kernels.cuh"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "cuda_utils.cuh"

__host__ __device__ __forceinline__ int32_t ceil_div(int32_t a, int32_t b) { return (a + b - 1) / b; }
constexpr int32_t __host__ __device__ ceil_div_static(int32_t a, int32_t b) { return (a + b - 1) / b; }

template <typename T>
__host__ __device__ __forceinline__ void swap_pointers(T** a, T** b) {
    auto temp_a = *a;
    *a = *b;
    *b = temp_a;
}

template <int tilesize, int numthreads>
__global__ void base_applyQt_singletile_evelyne( //aplies Qt (given by householder reflectors on diagonal tile k) to the remainder of the row
    int size_in,
    int size_out,
    int diag_iter,
    bool offsetdiag,
    float const *tau,
    float *in, float *out) {
    int g = blockIdx.x;
    int i = threadIdx.x;
    int j = threadIdx.y;
    __shared__ float outs[tilesize][tilesize];
    __shared__ float Qs[tilesize][tilesize];
    __shared__ float cache[tilesize][numthreads];
    int diagstartidx=diag_iter*tilesize;
    int tileoffset=(g)*tilesize;
    if (offsetdiag){
        tileoffset+=diagstartidx+tilesize;
    }


    for (int l=j;l<tilesize;l+=numthreads){
        outs[i][l]=out[(i+diagstartidx)*size_out+l+tileoffset];
        Qs[i][l]=in[(i+diagstartidx)*size_in+l+diagstartidx];
    }


    __syncthreads();

    for (int k=0;k<tilesize-1;k++){
        float tmp_sum = 0.0f;
        for (int l=k+j+1;l<tilesize;l+=numthreads){
            tmp_sum+= Qs[l][k]*outs[l][i];
        }
        cache[i][j]=tmp_sum;
        __syncthreads();
        tmp_sum=outs[k][i];
        for (int l=0;l<numthreads;l++){
            tmp_sum+=cache[i][l];
        }
        tmp_sum*=tau[(diag_iter)*size_in+k];
        for (int l=k+j+1;l<tilesize;l+=numthreads){
            outs[l][i]-=tmp_sum*Qs[l][k];
        }
        if (j==0){
            outs[k][i]-=tmp_sum;
        }
        __syncthreads();
    }

    for (int l=j;l<tilesize;l+=numthreads){
        out[(i+diagstartidx)*size_out+l+tileoffset]=outs[i][l];
    }
}

void launch_base_applyQt_singletile_evelyne(int size_in, int diag_iter, float const *tau, float *out) {
    const auto tilesize = 32;
    const auto numthreads = 4;

    // Need to launch one block for tile to the right of the diagonal to be processed
    const auto num_blocks = (size_in / tilesize - 1) - diag_iter;

    if (num_blocks <= 0)
        return;

    base_applyQt_singletile_evelyne<tilesize, numthreads><<<num_blocks, dim3(tilesize, numthreads)>>>(size_in, size_in, diag_iter, true, tau, out, out);
}

template <int tile_size>
__global__ void base_applyQt_singletile(int size_X, int row_stride_X, int row_stride_Q, float const *taus, float const* Q, float *X) {
    auto total_num_threads = gridDim.x * blockDim.x;
    auto block_num_threads = blockDim.x;
    auto columns_per_block = ceil_div(size_X, gridDim.x);

    const auto column_prefetch_size = 256;
    __shared__ float column_prefetch[tile_size][column_prefetch_size];

    // Each prefetch step consists of loading a chunk of columns from DRAM into shmem, and then processing them
    auto block_column_base_idx = blockIdx.x * columns_per_block;
    for (auto prefetch_step = 0; prefetch_step < ceil_div(columns_per_block, column_prefetch_size); prefetch_step++) {
        auto prefetch_column_base_idx = block_column_base_idx + prefetch_step * column_prefetch_size;

        for (auto prefetch_element_idx = threadIdx.x; prefetch_element_idx < column_prefetch_size * tile_size; prefetch_element_idx += block_num_threads) {
            auto local_prefetch_j = prefetch_element_idx % column_prefetch_size;
            auto local_prefetch_i = prefetch_element_idx / column_prefetch_size;
            auto global_prefetch_j = prefetch_column_base_idx + local_prefetch_j;
            auto global_prefetch_i = local_prefetch_i;
            column_prefetch[local_prefetch_i][local_prefetch_j] = global_prefetch_j < size_X ? X[global_prefetch_i * row_stride_X + global_prefetch_j] : 0.0f;
        }

        float current_column[tile_size];
        for (auto current_column_in_prefetch_step = 0; current_column_in_prefetch_step < ceil_div(column_prefetch_size, block_num_threads); current_column_in_prefetch_step++) {
            auto current_column_j_local = current_column_in_prefetch_step * block_num_threads + threadIdx.x;
            auto current_column_j = prefetch_column_base_idx + current_column_j_local;

            if (current_column_j >= size_X || current_column_j >= prefetch_column_base_idx + column_prefetch_size || current_column_j >= block_column_base_idx + columns_per_block)
                break;

            // Load current column we are processing
            for (auto local_i = 0; local_i < tile_size; local_i++) {
                auto current_column_i = local_i;
                current_column[local_i] = column_prefetch[current_column_i][current_column_j_local];
            }

            // Process current column by applying householder reflectors in reverse order
            float tau;
            float householder_reflector[tile_size];
            for (auto householder_reflector_idx = 0; householder_reflector_idx < tile_size; householder_reflector_idx++) {
                // for (auto i = householder_reflector_idx + 1; i < tile_size; i++) {
                tau = __ldg(&taus[householder_reflector_idx]);
                for (auto i = 0; i < tile_size; i++) {
                    householder_reflector[i] = __ldg(&Q[i * row_stride_Q + householder_reflector_idx]);
                }

                // First we compute tau * (h' x)
                auto effective_scaling = 0.0f;
                for (auto element_idx = 0; element_idx < tile_size; element_idx++) {
                    if (element_idx == householder_reflector_idx) {
                        // Implicit leading 1 in householder reflector
                        effective_scaling += current_column[element_idx];
                    } else if (element_idx > householder_reflector_idx) {
                        effective_scaling += householder_reflector[element_idx] * current_column[element_idx];
                    }
                }
                effective_scaling *= tau;

                // We now compute h (tau * (h' x)) to wrap things up
                for (auto element_idx = 0; element_idx < tile_size; element_idx++) {
                    if (element_idx == householder_reflector_idx) {
                        current_column[element_idx] -= effective_scaling;
                    } else if (element_idx > householder_reflector_idx) {
                        current_column[element_idx] -= effective_scaling * householder_reflector[element_idx];
                    }
                }
            }

            // Write out processed column
            for (auto local_i = 0; local_i < tile_size; local_i++) {
                auto current_column_i = local_i;
                X[current_column_i * row_stride_X + current_column_j] = current_column[local_i];
            }
        }
    }
}

void launch_base_applyQt_singletile(int size_in, int diag_iter, float const *tau, float *out) {
    const auto tilesize = 32;
    auto size_X = size_in - (diag_iter + 1) * tilesize;
    auto row_stride_Q = size_in;
    auto row_stride_X = size_in;
    auto Q = &out[diag_iter * tilesize * size_in + diag_iter * tilesize];
    auto X = &out[diag_iter * tilesize * size_in + (diag_iter + 1) * tilesize];
    auto taus = &tau[diag_iter * size_in];

    base_applyQt_singletile<tilesize><<<tilesize, 4 * tilesize>>>(size_X, row_stride_X, row_stride_Q, taus, Q, X);
}


void reference_applyQt(int size_in, int diag_iter, const float* tau, float* matrix) {
    cusolverDnHandle_t handle;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));

    int tilesize = 32;  // match the original implementation
    int diagstartidx = diag_iter * tilesize;

    const float* taus = &tau[diag_iter * size_in];

    // We need:
    // 1. The Householder vectors (from the diagonal tile)
    // 2. The part of the matrix we're updating

    // First, extract the Householder vectors from the diagonal tile
    float* householder_vectors;
    CHECK_CUDA(cudaMalloc(&householder_vectors, tilesize * tilesize * sizeof(float)));

    // Copy the diagonal tile (contains the Householder vectors)
    for(int j = 0; j < tilesize; j++) {
        CHECK_CUDA(cudaMemcpy(householder_vectors + j * tilesize,
                             matrix + (diagstartidx + j) * size_in + diagstartidx,
                             tilesize * sizeof(float),
                             cudaMemcpyDeviceToDevice));
    }

    // For each tile to the right of the diagonal
    for(int g = 1; g < (size_in - diagstartidx) / tilesize; g++) {
        // Extract the tile we're updating
        float* work_tile;
        CHECK_CUDA(cudaMalloc(&work_tile, tilesize * tilesize * sizeof(float)));
        for(int j = 0; j < tilesize; j++) {
            CHECK_CUDA(cudaMemcpy(work_tile + j * tilesize,
                                 matrix + (diagstartidx + j + g * tilesize) * size_in + diagstartidx,
                                 tilesize * sizeof(float),
                                 cudaMemcpyDeviceToDevice));
        }

        // Query workspace size
        int lwork;
        CHECK_CUSOLVER(cusolverDnSormqr_bufferSize(
            handle,
            CUBLAS_SIDE_LEFT,   // apply Q from the left
            CUBLAS_OP_T,        // apply Q transpose
            tilesize, tilesize, // dimensions of the tile
            tilesize,           // number of elementary reflectors
            householder_vectors, // the reflectors
            tilesize,           // leading dimension
            taus,               // tau values
            work_tile,          // matrix to update
            tilesize,           // leading dimension
            &lwork));

        // Allocate workspace
        float* workspace;
        int* devInfo;
        CHECK_CUDA(cudaMalloc(&workspace, lwork * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));

        // Apply Qt to the tile
        CHECK_CUSOLVER(cusolverDnSormqr(
            handle,
            CUBLAS_SIDE_LEFT,   // apply Q from the left
            CUBLAS_OP_T,        // apply Q transpose
            tilesize, tilesize, // dimensions of the tile
            tilesize,           // number of elementary reflectors
            householder_vectors, // the reflectors
            tilesize,           // leading dimension
            taus,               // tau values
            work_tile,          // matrix to update
            tilesize,           // leading dimension
            workspace, lwork, devInfo));

        // Copy result back
        for(int j = 0; j < tilesize; j++) {
            CHECK_CUDA(cudaMemcpy(matrix + (diagstartidx + j + g * tilesize) * size_in + diagstartidx,
                                 work_tile + j * tilesize,
                                 tilesize * sizeof(float),
                                 cudaMemcpyDeviceToDevice));
        }

        // Cleanup this iteration
        CHECK_CUDA(cudaFree(workspace));
        CHECK_CUDA(cudaFree(devInfo));
        CHECK_CUDA(cudaFree(work_tile));
    }

    // Final cleanup
    CHECK_CUDA(cudaFree(householder_vectors));
    CHECK_CUSOLVER(cusolverDnDestroy(handle));
}

struct reference_applyQt_fast_workspace {
    cusolverDnHandle_t handle;
    float* householder_vectors;
    float* input;
    const float* taus;
    float* workspace;
    int* devInfo;
    float* matrix_col_major;
    int lwork;
};

void* reference_applyQt_fast_preamble(int size_in, int diag_iter, const float* tau, float* matrix) {
    float* matrix_col_major;
    CHECK_CUDA(cudaMalloc(&matrix_col_major, size_in * size_in * sizeof(float)));
    convert_matrix_major(matrix, matrix_col_major, size_in, size_in);

    int tilesize = 32;  // match the original implementation
    int diagstartidx = diag_iter * tilesize;

    cusolverDnHandle_t handle;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));

    const float* taus = &tau[diag_iter * size_in];

    // Copy the diagonal tile (contains the Householder vectors)
    float* householder_vectors;
    CHECK_CUDA(cudaMalloc(&householder_vectors, tilesize * tilesize * sizeof(float)));

    for(int j = 0; j < tilesize; j++) {
        CHECK_CUDA(cudaMemcpyAsync(householder_vectors + j * tilesize,
                             matrix_col_major + (diagstartidx + j) * size_in + diagstartidx,
                             tilesize * sizeof(float),
                             cudaMemcpyDeviceToDevice));
    }

    // Copy everything to the right of the diagonal tile
    float* input;
    CHECK_CUDA(cudaMalloc(&input, (size_in - diagstartidx - tilesize) * tilesize * sizeof(float)));
    for(int j = 0; j < size_in - diagstartidx - tilesize; j++) {
        CHECK_CUDA(cudaMemcpyAsync(input + j * tilesize,
                                 matrix_col_major + (diagstartidx + j + tilesize) * size_in + diagstartidx,
                                 tilesize * sizeof(float),
                                 cudaMemcpyDeviceToDevice));
    }

    // Allocate workspace
    int lwork;
    CHECK_CUSOLVER(cusolverDnSormqr_bufferSize(
        handle,
        CUBLAS_SIDE_LEFT,   // apply Q from the left
        CUBLAS_OP_T,        // apply Q transpose
        tilesize, size_in - diagstartidx - tilesize, // dimensions of the tile
        tilesize,           // number of elementary reflectors
        householder_vectors, // the reflectors
        tilesize,           // leading dimension
        taus,               // tau values
        input,          // matrix to update
        tilesize,        // leading dimension
        &lwork));

    // Allocate workspace
    float* workspace;
    int* devInfo;
    CHECK_CUDA(cudaMalloc(&workspace, lwork * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));

    reference_applyQt_fast_workspace* kernel_workspace = new reference_applyQt_fast_workspace();
    kernel_workspace->handle = handle;
    kernel_workspace->householder_vectors = householder_vectors;
    kernel_workspace->input = input;
    kernel_workspace->workspace = workspace;
    kernel_workspace->devInfo = devInfo;
    kernel_workspace->matrix_col_major = matrix_col_major;
    kernel_workspace->taus = taus;
    kernel_workspace->lwork = lwork;
    return kernel_workspace;
}

void reference_applyQt_fast(int size_in, int diag_iter, const float* tau, float* matrix, void* workspace_ptr) {
    reference_applyQt_fast_workspace* workspace = static_cast<reference_applyQt_fast_workspace*>(workspace_ptr);

    int tilesize = 32;  // match the original implementation
    int diagstartidx = diag_iter * tilesize;

    const float* taus = workspace->taus;

    CHECK_CUSOLVER(cusolverDnSormqr(
        workspace->handle,
        CUBLAS_SIDE_LEFT,   // apply Q from the left
        CUBLAS_OP_T,        // apply Q transpose
        tilesize, size_in - diagstartidx - tilesize, // dimensions of the tile
        tilesize,           // number of elementary reflectors
        workspace->householder_vectors, // the reflectors
        tilesize,           // leading dimension
        taus,               // tau values
        workspace->input,          // matrix to update
        tilesize,        // leading dimension
        workspace->workspace, workspace->lwork, workspace->devInfo));
}

void reference_applyQt_fast_postamble(int size_in, int diag_iter, const float* tau, float* matrix, void* workspace_ptr) {
    int tilesize = 32;  // match the original implementation
    int diagstartidx = diag_iter * tilesize;

    reference_applyQt_fast_workspace* workspace = static_cast<reference_applyQt_fast_workspace*>(workspace_ptr);

    // Copy input back into matrix
    for(int j = 0; j < size_in - diagstartidx - tilesize; j++) {
        CHECK_CUDA(cudaMemcpyAsync(workspace->matrix_col_major + (diagstartidx + j + tilesize) * size_in + diagstartidx,
                                 workspace->input + j * tilesize,
                                 tilesize * sizeof(float),
                                 cudaMemcpyDeviceToDevice));
    }

    convert_matrix_major(workspace->matrix_col_major, matrix, size_in, size_in, false);
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaFree(workspace->input));
    CHECK_CUDA(cudaFree(workspace->householder_vectors));
    CHECK_CUDA(cudaFree(workspace->workspace));
    CHECK_CUDA(cudaFree(workspace->devInfo));
    CHECK_CUDA(cudaFree(workspace->matrix_col_major));
    CHECK_CUSOLVER(cusolverDnDestroy(workspace->handle));
    delete workspace;
}
