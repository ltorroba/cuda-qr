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
__global__ void base_applyQt_singletile(int size_X, int row_stride_X, int row_stride_Q, float const *tau, float const* Q, float *X) {
    auto num_threads = gridDim.x * blockDim.x;
    auto thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto columns_per_thread = ceil_div(size_X, num_threads);

    // TODO: Use all threads in block to load the columns we will be processing using a better
    //       access pattern

    // Load householder reflectors and taus
    float householder_reflectors[tile_size][tile_size];
    float taus[tile_size];
    for (auto reflector_idx = 0; reflector_idx < tile_size; reflector_idx++) {
        taus[reflector_idx] = tau[reflector_idx];
        for (auto element_idx = reflector_idx + 1; element_idx < tile_size; element_idx++) {
            auto householder_reflector_i = element_idx;
            auto householder_reflector_j = reflector_idx;
            householder_reflectors[element_idx][reflector_idx] = Q[householder_reflector_i * row_stride_Q + householder_reflector_j];
        }
    }

    float current_column[tile_size];
    for (auto current_column_idx = 0; current_column_idx < columns_per_thread; current_column_idx++) {
        auto current_column_j = thread_idx * columns_per_thread + current_column_idx;

        if (current_column_j >= size_X)
            break;

        // Load current column we are processing
        for (auto local_i = 0; local_i < tile_size; local_i++) {
            auto current_column_i = local_i;
            current_column[local_i] = X[current_column_i * row_stride_X + current_column_j];
        }

        // Process current column by applying householder reflectors in reverse order
        for (auto householder_reflector = 0; householder_reflector < tile_size; householder_reflector++) {
            // First we compute tau * (h' x)
            auto effective_scaling = 0.0f;
            for (auto element_idx = 0; element_idx < tile_size; element_idx++) {
                if (element_idx == householder_reflector) {
                    // Implicit leading 1 in householder reflector
                    effective_scaling += current_column[element_idx];
                } else if (element_idx > householder_reflector) {
                    effective_scaling += householder_reflectors[element_idx][householder_reflector] * current_column[element_idx];
                }
            }
            effective_scaling *= taus[householder_reflector];

            // We now compute h (tau * (h' x)) to wrap things up
            for (auto element_idx = 0; element_idx < tile_size; element_idx++) {
                if (element_idx == householder_reflector) {
                    current_column[element_idx] -= effective_scaling;
                } else if (element_idx > householder_reflector) {
                    current_column[element_idx] -= effective_scaling * householder_reflectors[element_idx][householder_reflector];
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

void launch_base_applyQt_singletile(int size_in, int diag_iter, float const *tau, float *out) {
    const auto tilesize = 32;
    auto size_X = size_in - (diag_iter + 1) * tilesize;
    auto row_stride_Q = size_in;
    auto row_stride_X = size_in;
    auto Q = &out[diag_iter * tilesize * size_in + diag_iter * tilesize];
    auto X = &out[diag_iter * tilesize * size_in + (diag_iter + 1) * tilesize];
    auto taus = &tau[diag_iter * size_in];

    base_applyQt_singletile<tilesize><<<tilesize, tilesize>>>(size_X, row_stride_X, row_stride_Q, taus, Q, X);
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


void reference_applyQt_fast(int size_in, int diag_iter, const float* tau, float* matrix) {
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
        CHECK_CUDA(cudaMemcpyAsync(householder_vectors + j * tilesize,
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