#include "reference_kernels.cuh"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "cuda_utils.cuh"

template <int tilesize, int numthreads>
__global__ void base_applyQt_singletile( //aplies Qt (given by householder reflectors on diagonal tile k) to the remainder of the row
    int size_in,
    int diag_iter,
    float const *tau,
    float *out) {
    int g = blockIdx.x;
    int i = threadIdx.x;
    int j = threadIdx.y;
    __shared__ float outs[tilesize][tilesize];
    __shared__ float Qs[tilesize][tilesize];
    __shared__ float cache[tilesize][numthreads];
    int diagstartidx=diag_iter*tilesize;
    int tileoffset=(1+g)*tilesize;
    
    
    for (int l=j;l<tilesize;l+=numthreads){
        outs[i][l]=out[(i+diagstartidx)*size_in+l+diagstartidx+tileoffset];
        Qs[i][l]=out[(i+diagstartidx)*size_in+l+diagstartidx];
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
        out[(i+diagstartidx)*size_in+l+diagstartidx+tileoffset]=outs[i][l];
    }
}


void reference_applyQt(int size_in, int diag_iter, const float* tau, float* matrix) {
    cusolverDnHandle_t handle;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));
    
    int tilesize = 32;  // match the original implementation
    int diagstartidx = diag_iter * tilesize;
    
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
        int tileoffset = g * tilesize;
        
        // Extract the tile we're updating
        float* work_tile;
        CHECK_CUDA(cudaMalloc(&work_tile, tilesize * tilesize * sizeof(float)));
        for(int j = 0; j < tilesize; j++) {
            CHECK_CUDA(cudaMemcpy(work_tile + j * tilesize,
                                 matrix + (diagstartidx + j) * size_in + diagstartidx + tileoffset,
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
            tau + diagstartidx, // tau values
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
            tau + diagstartidx, // tau values
            work_tile,          // matrix to update
            tilesize,           // leading dimension
            workspace, lwork, devInfo));
            
        // Copy result back
        for(int j = 0; j < tilesize; j++) {
            CHECK_CUDA(cudaMemcpy(matrix + (diagstartidx + j) * size_in + diagstartidx + tileoffset,
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

// After all the other code, add this explicit instantiation:
template __global__ void base_applyQt_singletile<32, 32>(int size_in, 
                                                        int diag_iter, 
                                                        const float* tau, 
                                                        float* out);