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
    const auto numthreads = 1;

    // Need to launch one block for tile to the right of the diagonal to be processed
    const auto num_blocks = (size_in / tilesize - 1) - diag_iter;

    if (num_blocks <= 0)
        return;

    base_applyQt_singletile_evelyne<tilesize, numthreads><<<num_blocks, dim3(tilesize, numthreads)>>>(size_in, size_in, diag_iter, true, tau, out, out);
}

template <int tile_size, int columns_per_block, int threads_per_block>
__global__ void base_applyQt_singletile(int size_X, int row_stride_X, int row_stride_Q, float const *taus, float const* Q, float *X) {
    // TODO: Static assert that tile_size % microtile_size == 0
    auto const block_num_threads = threads_per_block;
    const auto microtile_size = columns_per_block / threads_per_block;
    const auto column_prefetch_size = columns_per_block;
    __shared__ float column_prefetch_1[tile_size * column_prefetch_size];
    // __shared__ float column_prefetch_2[tile_size * column_prefetch_size];
    float* column_prefetch = column_prefetch_1;
    // float* column_prefetch_backup = column_prefetch_2;

    // Each prefetch step consists of loading a chunk of columns from DRAM into shmem, and then processing them
    auto block_column_base_idx = blockIdx.x * columns_per_block;
    for (auto prefetch_step = 0; prefetch_step < ceil_div_static(columns_per_block, column_prefetch_size); prefetch_step++) {
        auto prefetch_column_base_idx = block_column_base_idx + prefetch_step * column_prefetch_size;

        for (auto prefetch_element_idx = threadIdx.x; prefetch_element_idx < column_prefetch_size * tile_size; prefetch_element_idx += block_num_threads) {
            auto local_prefetch_j = prefetch_element_idx % column_prefetch_size;
            auto local_prefetch_i = prefetch_element_idx / column_prefetch_size;
            auto global_prefetch_j = prefetch_column_base_idx + local_prefetch_j;
            auto global_prefetch_i = local_prefetch_i;
            column_prefetch[local_prefetch_i * column_prefetch_size + local_prefetch_j] = global_prefetch_j < size_X ? X[global_prefetch_i * row_stride_X + global_prefetch_j] : 0.0f;
        }

        auto early_exit = false;
        for (auto current_microtile_in_prefetch_step = 0; current_microtile_in_prefetch_step < ceil_div_static(column_prefetch_size / microtile_size, block_num_threads); current_microtile_in_prefetch_step++) {
            if (early_exit) break;

            for (auto current_column_in_microtile = 0; current_column_in_microtile < microtile_size; current_column_in_microtile++) {
                float current_column[tile_size];

                auto current_column_j_local = (current_microtile_in_prefetch_step * block_num_threads + threadIdx.x) * microtile_size + current_column_in_microtile;
                auto current_column_j = prefetch_column_base_idx + current_column_j_local;

                if (current_column_j >= size_X) {
                    early_exit = true;
                    break;
                }

                // Load current column we are processing
                for (auto local_i = 0; local_i < tile_size; local_i++) {
                    auto current_column_i = local_i;
                    current_column[local_i] = column_prefetch[current_column_i * column_prefetch_size + current_column_j_local];
                }

                // Process current column by applying householder reflectors
                for (auto householder_reflector_idx = 0; householder_reflector_idx < tile_size; householder_reflector_idx++) {
                    float householder_reflector[tile_size];
                    float tau = taus[householder_reflector_idx];

                    for (auto i = householder_reflector_idx; i < tile_size; i++) {
                        householder_reflector[i] = Q[i * row_stride_Q + householder_reflector_idx];
                    }

                    // First we compute tau * (h' x)
                    auto effective_scaling = current_column[householder_reflector_idx];
                    for (auto element_idx = householder_reflector_idx + 1; element_idx < tile_size; element_idx++) {
                        effective_scaling += householder_reflector[element_idx] * current_column[element_idx];
                    }
                    effective_scaling *= tau;

                    // We now compute h (tau * (h' x)) to wrap things up
                    current_column[householder_reflector_idx] -= effective_scaling;
                    for (auto element_idx = householder_reflector_idx + 1; element_idx < tile_size; element_idx++) {
                        current_column[element_idx] -= effective_scaling * householder_reflector[element_idx];
                    }

                    // We actually know that all elements <= householder_reflector_idx will remain unchanged from now on,
                    // so we can commit these changes to global memory already
                    X[householder_reflector_idx * row_stride_X + current_column_j] = current_column[householder_reflector_idx];
                }
            }
        }

        // swap_pointers(&column_prefetch, &column_prefetch_backup);
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

    // const auto num_blocks = (size_in / tilesize - 1) - diag_iter;
    const auto columns_per_block = 32;
    const auto threads_per_block = 32;
    auto num_blocks = ceil_div(size_X, columns_per_block);
    base_applyQt_singletile<tilesize, columns_per_block, threads_per_block><<<num_blocks, threads_per_block>>>(size_X, row_stride_X, row_stride_Q, taus, Q, X);
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

//////// TENSOR CORE IMPLEMENTATION

__device__ float get_reflector_coordinate(int reflector_idx, int coordinate_idx, int row_stride_Q, float const* Q) {
    return coordinate_idx > reflector_idx ? Q[coordinate_idx * row_stride_Q + reflector_idx] : (coordinate_idx == reflector_idx ? 1.0f : 0.0f);
}

template <int threads_per_block, int num_householder_vectors, int vector_dim>
__device__ void materialize_reflector_pair_async(int reflector_L, int reflector_R, int row_stride_Q, float const* taus, float const* Q, float const* alphas, float* out) {
    // TODO: Optimizations
    //      - We can assume that reflector_1 < reflector_2 always
    //      - Since reflectors are adjacent, we can probably save compute
    for (auto element_idx = threadIdx.x; element_idx < vector_dim * vector_dim; element_idx += threads_per_block) {
        auto i_idx = element_idx / vector_dim;
        auto j_idx = element_idx % vector_dim;

        auto tau_L = taus[reflector_L];
        auto tau_R = taus[reflector_R];

        auto reflector_L_i = get_reflector_coordinate(reflector_L, i_idx, row_stride_Q, Q);
        auto reflector_L_j = get_reflector_coordinate(reflector_L, j_idx, row_stride_Q, Q);
        auto reflector_R_i = get_reflector_coordinate(reflector_R, i_idx, row_stride_Q, Q);
        auto reflector_R_j = get_reflector_coordinate(reflector_R, j_idx, row_stride_Q, Q);

        // TODO: Probably better to just materialize alpha on the fly
        auto alpha_LR = alphas[reflector_L * num_householder_vectors + reflector_R];

        // Dirac (i.e. identity-matrix component) along diagonal
        auto accumulator = i_idx == j_idx ? 1.0f : 0.0f;

        // reflector_L term
        accumulator -= tau_L * reflector_L_i * reflector_L_j;

        // reflector_R term
        accumulator -= tau_R * reflector_R_i * reflector_R_j;

        // cross term
        accumulator -= tau_L * tau_R * alpha_LR * reflector_L_i * reflector_R_j;

        out[i_idx * vector_dim + j_idx] = accumulator;
    }
}

template <int tile_size, int threads_per_block>
__device__ void tile_multiply_accumulate_async(int row_stride_tile_A, int row_stride_tile_B, int row_stride_tile_C, float* tile_A, float* tile_B, float* tile_C) {
    constexpr auto num_warps_in_block = threads_per_block / 32;
    constexpr auto subtiles_per_row = tile_size / 8;
    const auto warp_idx = threadIdx.x / 32;
    auto tidx = threadIdx.x % 32;

    for (auto subtile_idx = warp_idx; subtile_idx < tile_size * tile_size / (16 * 8); subtile_idx += num_warps_in_block) {
        // Calculate indices of the 16 x 8 subtile of the product AB that we will be responsible for computing here
        auto subtile_i = subtile_idx / subtiles_per_row;
        auto subtile_j = subtile_idx % subtiles_per_row;

        // These are the indices of the 16 x 8 warptile we are currently processing
        // that this thread will emit
        const int32_t c_1_local_idx = 2 * tidx;
        const int32_t c_2_local_idx = 2 * tidx + 1;
        const int32_t c_3_local_idx = c_1_local_idx + 64;
        const int32_t c_4_local_idx = c_2_local_idx + 64;

        // Create accumulators for final result
        float c_1 = 0.0f;
        float c_2 = 0.0f;
        float c_3 = 0.0f;
        float c_4 = 0.0f;

        for (auto k_idx = 0; k_idx < tile_size / 8; k_idx++) {
            // LOAD THE CORRECT 16X8 FRAGMENT OF A
            // Compute the indices of the 16 x 8 slice of A that need to be pulled for the tensor core computation
            int32_t a_1_local_idx = 2 * 4 * (tidx / 4) + tidx % 4;
            int32_t a_2_local_idx = a_1_local_idx + 64;
            int32_t a_3_local_idx = a_1_local_idx + 4;
            int32_t a_4_local_idx = a_2_local_idx + 4;

            // Compute what those indices correspond to in tile_A [basically (local_i_coord) * row_stride_tile_A + (local_k_coord)]
            int32_t a_1_tile_idx = (subtile_i * 16 + a_1_local_idx / 8) * row_stride_tile_A + (k_idx * 8 + a_1_local_idx % 8);
            int32_t a_2_tile_idx = (subtile_i * 16 + a_2_local_idx / 8) * row_stride_tile_A + (k_idx * 8 + a_2_local_idx % 8);
            int32_t a_3_tile_idx = (subtile_i * 16 + a_3_local_idx / 8) * row_stride_tile_A + (k_idx * 8 + a_3_local_idx % 8);
            int32_t a_4_tile_idx = (subtile_i * 16 + a_4_local_idx / 8) * row_stride_tile_A + (k_idx * 8 + a_4_local_idx % 8);

            float a_1 = tile_A[a_1_tile_idx];
            float a_2 = tile_A[a_2_tile_idx];
            float a_3 = tile_A[a_3_tile_idx];
            float a_4 = tile_A[a_4_tile_idx];

            // LOAD TILE B
            // Compute the indices of the 8 x 8 microtile of B that need to be pulled for the tensor core computation
            int32_t b_1_local_idx = (tidx * 8) % 32 + tidx / 4;
            int32_t b_2_local_idx = b_1_local_idx + 32;

            // Compute what those indices correspond to in the SMEM tile of B [basically (local_k_coord) * tile_size_j + (local_j_coord)]
            int32_t b_1_tile_idx = (k_idx * 8 + b_1_local_idx / 8) * row_stride_tile_B + (subtile_j * 8 + b_1_local_idx % 8);
            int32_t b_2_tile_idx = (k_idx * 8 + b_2_local_idx / 8) * row_stride_tile_B + (subtile_j * 8 + b_2_local_idx % 8);

            float b_1 = tile_B[b_1_tile_idx];
            float b_2 = tile_B[b_2_tile_idx];

            uint32_t ua_1 = __float_as_uint(a_1);
            uint32_t ua_2 = __float_as_uint(a_2);
            uint32_t ua_3 = __float_as_uint(a_3);
            uint32_t ua_4 = __float_as_uint(a_4);

            uint32_t ub_1 = __float_as_uint(b_1);
            uint32_t ub_2 = __float_as_uint(b_2);

            uint32_t uc_1 = __float_as_uint(c_1);
            uint32_t uc_2 = __float_as_uint(c_2);
            uint32_t uc_3 = __float_as_uint(c_3);
            uint32_t uc_4 = __float_as_uint(c_4);

            asm(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
                "{%0, %1, %2, %3},     /* 'D' matrix */"
                "{%4, %5, %6, %7},     /* 'A' matrix */"
                "{%8, %9},             /* 'B' matrix */"
                "{%10, %11, %12, %13}; /* 'C' matrix */"
                : "=r"(uc_1), "=r"(uc_2), "=r"(uc_3), "=r"(uc_4)
                : "r"(ua_1), "r"(ua_2), "r"(ua_3), "r"(ua_4) , "r"(ub_1), "r"(ub_2), "r"(uc_1), "r"(uc_2), "r"(uc_3), "r"(uc_4)
            );

            c_1 = __uint_as_float(uc_1);
            c_2 = __uint_as_float(uc_2);
            c_3 = __uint_as_float(uc_3);
            c_4 = __uint_as_float(uc_4);
        }

        int32_t c_1_global_idx = (subtile_i * 16 + (c_1_local_idx / 8)) * row_stride_tile_C + (subtile_j * 8 + c_1_local_idx % 8);
        int32_t c_2_global_idx = (subtile_i * 16 + (c_2_local_idx / 8)) * row_stride_tile_C + (subtile_j * 8 + c_2_local_idx % 8);
        int32_t c_3_global_idx = (subtile_i * 16 + (c_3_local_idx / 8)) * row_stride_tile_C + (subtile_j * 8 + c_3_local_idx % 8);
        int32_t c_4_global_idx = (subtile_i * 16 + (c_4_local_idx / 8)) * row_stride_tile_C + (subtile_j * 8 + c_4_local_idx % 8);

        tile_C[c_1_global_idx] = c_1;
        tile_C[c_2_global_idx] = c_2;
        tile_C[c_3_global_idx] = c_3;
        tile_C[c_4_global_idx] = c_4;
    }
}

template <int tile_size, int threads_per_block>
__global__ void base_applyQt_singletile_tc(int size_X, int row_stride_X, int row_stride_Q, float const *taus, float const* Q, float *X) {
    __shared__ float alphas[tile_size * tile_size];

    // We assume that we have tile_size vectors each with (implict) dimensionality tile_size
    auto const num_householder_vectors = tile_size;
    auto const vector_dim = tile_size;

    __shared__ float tile_buffer_1[vector_dim * vector_dim];
    __shared__ float tile_buffer_2[vector_dim * vector_dim];
    __shared__ float tile_buffer_3[vector_dim * vector_dim];

    float* reflector_matrix = tile_buffer_1;
    float* tile_buffer = tile_buffer_2;
    float* tile_buffer_backup = tile_buffer_3;

    // Compute alphas across a block
    // TODO: Optimizations:
    //    - alpha is symmetric, so only need to compute lower triangle
    //    - can probably improve things by forcing elements to be consecutive, which means compiler will do register reuse
    for (auto element_idx = threadIdx.x; element_idx < num_householder_vectors * num_householder_vectors; element_idx += threads_per_block) {
        auto i_idx = element_idx / num_householder_vectors;
        auto j_idx = element_idx % num_householder_vectors;

        auto result = 0.0f;
        for (auto coordinate = 0; coordinate < vector_dim; coordinate++) {
            auto i_coord_val = get_reflector_coordinate(i_idx, coordinate, row_stride_Q, Q);
            auto j_coord_val = get_reflector_coordinate(j_idx, coordinate, row_stride_Q, Q);
            result += i_coord_val * j_coord_val;
        }

        alphas[i_idx * num_householder_vectors + j_idx] = result;
    }

    __syncthreads();

    // Materialize matrix for first pair of reflectors
    materialize_reflector_pair_async<threads_per_block, num_householder_vectors, vector_dim>(1, 0, row_stride_Q, taus, Q, alphas, reflector_matrix);

    for (auto pair_idx = 1; pair_idx < tile_size / 2; pair_idx++) {
        materialize_reflector_pair_async<threads_per_block, num_householder_vectors, vector_dim>(2 * pair_idx + 1, 2 * pair_idx, row_stride_Q, taus, Q, alphas, tile_buffer);
        __syncthreads();
        tile_multiply_accumulate_async<threads_per_block, tile_size>(tile_size, tile_size, tile_size, tile_buffer, reflector_matrix, tile_buffer_backup);
        swap_pointers(&tile_buffer_backup, &reflector_matrix);
        __syncthreads();
    }

    // Perform the actual matmul
    for (auto tile_idx = blockIdx.x; tile_idx < ceil_div(size_X, tile_size); tile_idx += gridDim.x) {
        // Load tile into shared memory
        for (auto element_idx = threadIdx.x; element_idx < tile_size * tile_size; element_idx += threads_per_block) {
            auto i_idx = element_idx / tile_size;
            auto j_idx = element_idx % tile_size;
            tile_buffer[i_idx * tile_size + j_idx] = X[i_idx * row_stride_X + (tile_idx * tile_size + j_idx)];
        }
        __syncthreads();

        // Perform tile matmul
        tile_multiply_accumulate_async<threads_per_block, tile_size>(tile_size, tile_size, tile_size, reflector_matrix, tile_buffer, tile_buffer_2);
        __syncthreads();

        // Store output tile back to global memory
        for (auto element_idx = threadIdx.x; element_idx < tile_size * tile_size; element_idx += threads_per_block) {
            auto i_idx = element_idx / tile_size;
            auto j_idx = element_idx % tile_size;
            X[i_idx * row_stride_X + (tile_idx * tile_size + j_idx)] = tile_buffer_2[i_idx * tile_size + j_idx];
        }
        __syncthreads();
    }
}

void launch_base_applyQt_singletile_tc(int size_in, int diag_iter, float const *tau, float *out) {
    const auto tilesize = 32;
    auto size_X = size_in - (diag_iter + 1) * tilesize;
    auto row_stride_Q = size_in;
    auto row_stride_X = size_in;
    auto Q = &out[diag_iter * tilesize * size_in + diag_iter * tilesize];
    auto X = &out[diag_iter * tilesize * size_in + (diag_iter + 1) * tilesize];
    auto taus = &tau[diag_iter * size_in];

    auto num_blocks = ceil_div(size_X, 32);
    auto const threads_per_block = 32;
    base_applyQt_singletile_tc<tilesize, threads_per_block><<<num_blocks, threads_per_block>>>(size_X, row_stride_X, row_stride_Q, taus, Q, X);
}

