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
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <sstream>


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
__device__ inline void cp_async1(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 4;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.ca.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES));
}

__device__ __forceinline__ void async_memcpy_waitall() {
    asm volatile("cp.async.wait_all;\n" ::);
}

// Macro for checking CUDA errors
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

// Macro for checking cuBLAS errors
#define CHECK_CUBLAS(call) { \
    const cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error: %s:%d, status: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

// Macro for checking cuSOLVER errors
#define CHECK_CUSOLVER(call) { \
    const cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        printf("cuSOLVER Error: %s:%d, status: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

void print_matrix(int32_t n_row, int32_t n_col, std::vector<float> const &matrix) {
    for (int32_t i = 0; i < n_row; i++) {
        printf("    ");
        for (int32_t j = 0; j < n_col; j++) {
            printf("%5.2f ", matrix.at(i * n_col + j));
        }
        printf("\n");
    }
}

////////////////////////////////////////////////////////////////////////////////




namespace qr_multi {

    #define microtilesize 2
    #define tilesize 4
    #define nummicrotiles 2
    #define numilpqr 1
    #define numthreadsqr 2
    #define numthreadsqr2 2 
    #define nummicroblocksqr 2 //number of microblocks of numthreadsqr launched, >=2

    
    
    //-------------------- multiply by Q kernels-----------------------------------
    
    
    
    __global__ void multi_applyQt_singletile( //aplies Qt (given by householder reflectors on diagonal tile k) to the remainder of the row
        int size_in,
        int size_out,
        int diag_iter,
        bool offsetdiag,
        float const *tau,
        float *in, float *out) {
        int g = blockIdx.x;
        int i = threadIdx.x;
        float outs[microtilesize];
        __shared__ float Qs[microtilesize];
        int diagstartidx=diag_iter*microtilesize;
        int tileoffset=(g)*microtilesize;
        if (offsetdiag){
            tileoffset+=diagstartidx+microtilesize;
        }
        
        
        for (int l=0;l<microtilesize;l++){
            outs[l]=out[(l+diagstartidx)*size_out+i+tileoffset];
        }
        
    
        __syncthreads();
    
        for (int k=0;k<microtilesize-1;k++){
            Qs[i]=in[(i+diagstartidx)*size_in+k+diagstartidx];
            __syncthreads();
            float tmp_sum = 0.0f;
            for (int l=k+1;l<microtilesize;l++){
                tmp_sum+= Qs[l]*outs[l];
            }
            tmp_sum+=outs[k];
            tmp_sum*=tau[(diag_iter)*size_in+k];
            for (int l=k+1;l<microtilesize;l++){
                outs[l]-=tmp_sum*Qs[l];
            }
            outs[k]-=tmp_sum;
            __syncthreads();
        }
    
        for (int l=0;l<microtilesize;l++){
            out[(l+diagstartidx)*size_out+i+tileoffset]=outs[l];
        }
    
    }
    
    __global__ void multi_applyQt_doubletile( //aplies Qt (given by householder reflectors on the tile at row_idx below diag_idx) to the remainder of the row, and to the row of diag_idx
        int size_in,
        int size_out,
        int diag_iter,
        int row_iter,
        bool offsetdiag,
        float const *tau,
        float *in, float *out) {
        int g = blockIdx.x;
        int i = threadIdx.x;
        float outs[2*microtilesize];
        __shared__ float Qs[microtilesize];
        int diagstartidx=diag_iter*microtilesize;
        int tileoffset=(g)*microtilesize;
        int iteroffset=row_iter*microtilesize;
        if (offsetdiag){
            tileoffset+=diagstartidx+microtilesize;
        }
    
        for (int l=0;l<microtilesize;l++){
            outs[l]=out[(l+diagstartidx)*size_out+i+tileoffset];
            outs[l+microtilesize]=out[(l+diagstartidx+iteroffset)*size_out+i+tileoffset];
        }
    
        __syncthreads();
    
    
        for (int k=0;k<microtilesize;k++){
            Qs[i]=in[(i+diagstartidx+iteroffset)*size_in+k+diagstartidx];
            __syncthreads();
            float tmp_sum = 0.0f;
            for (int l=0;l<microtilesize;l++){
                tmp_sum+= Qs[l]*outs[l+microtilesize];
            }
            tmp_sum+=outs[k] ;
            tmp_sum*=tau[(diag_iter)*size_in+row_iter*microtilesize+k];
            outs[k]-=tmp_sum;
            for (int l=0;l<microtilesize;l++){
                outs[l+microtilesize]-=tmp_sum*Qs[l];
            }
            
            __syncthreads();
        }
    
        for (int l=0;l<microtilesize;l++){
            out[(l+diagstartidx)*size_out+i+tileoffset]=outs[l];
            out[(l+diagstartidx+iteroffset)*size_out+i+tileoffset]=outs[l+microtilesize];
        }
    
    }
    //-------------------- calculate QR kernels-----------------------------------
    
    
    
    __global__ __launch_bounds__(numthreadsqr*nummicroblocksqr) void multi_calcQR_singletile( //calculates in-place QR of diagonal tile
        int size_in,
        int diag_iter,
        float *tau,
        float *out) {
            
        int i = threadIdx.x;
        int i2 = threadIdx.y;
        float outs[2*microtilesize][nummicrotiles];
        __shared__ float cache[microtilesize];
        __shared__ float cache2[2];
        int diagstartidx=diag_iter*microtilesize*nummicrotiles;

        
        
        __syncthreads();

        for (int microiter=0; microiter<nummicrotiles; microiter++){
            int microdiagoffset=microiter*microtilesize;
            for (int microrow=0;microrow<nummicrotiles;microrow++){
                
                int microrowoffset=microrow*microtilesize;
                int tileoffset= (microrow==0 ? 0 : 1);
                
                
                if (i2==0){
                for (int k=0; k<microtilesize;k++){
                        for (int j=i; j<microtilesize; j+=numthreadsqr){
                            outs[k+(microrow!=0)*microtilesize][j/numthreadsqr]=out[(k+diagstartidx+microdiagoffset+microrowoffset)*size_in+j+diagstartidx+microdiagoffset];
                        }
                    }  
                }else{
                    for (int k=0; k<microtilesize;k++){
                        for (int j=(1+microiter)*microtilesize+i, l=0; j<microtilesize*nummicrotiles; j+=numthreadsqr*(nummicroblocksqr-1), l++){
                            outs[k+(microrow!=0)*microtilesize][l]=out[(k+diagstartidx+microdiagoffset+microrowoffset)*size_in+j+diagstartidx+microdiagoffset];
                        }
                    }
                }
                
                __syncthreads();
                for(int iter=0;iter<microtilesize-(microrow==0);iter++){
                    int startidxk=(microrow==0 ? iter+1 : 0);
                    float tmp_sum[numilpqr]={0.0f};
                    if (i==(iter%numthreadsqr) && i2==0){
                        for (int k= startidxk;k<microtilesize;k++){
                            cache[k]=outs[k+tileoffset][iter/numthreadsqr];
                            tmp_sum[iter/numthreadsqr]+=outs[k+tileoffset][iter/numthreadsqr]*outs[k+tileoffset][iter/numthreadsqr];
                        }
                        cache2[0]=tmp_sum[iter/numthreadsqr];
                        cache2[1]=outs[iter][iter/numthreadsqr];
                    }

                    __syncthreads();
                    if (i2==0){
                        for (int j=i; j<microtilesize; j+=numthreadsqr){
                            if (j>=iter){
                                float newvalue=cache2[1] + (cache2[1]>0 ? 1 : -1) * sqrt(cache2[0]+cache2[1]*cache2[1]); //u1
                                float taut=2 /(cache2[0]/(newvalue* newvalue)+1);
                                if (j>iter){
                                    for (int k=startidxk;k<microtilesize;k++){
                                        tmp_sum[j/numthreadsqr]+=cache[k]*outs[k+tileoffset][j/numthreadsqr];
                                    }
                                    tmp_sum[j/numthreadsqr]/= newvalue;
                                }else {
                                    tau[(diag_iter)*size_in+microtilesize*microrow+j]=taut;
                                }
                                
                                float tmp_sum2 = (tmp_sum[j/numthreadsqr] +outs[iter][j/numthreadsqr])*taut;
                                outs[iter][j/numthreadsqr]-=tmp_sum2;
                                if (j>iter){
                                    for (int k=startidxk;k<microtilesize;k++){
                                        outs[k+tileoffset][j/numthreadsqr]*=newvalue;
                                        outs[k+tileoffset][j/numthreadsqr]-=cache[k]* tmp_sum2;
                                    }
                                }
                                for (int k=startidxk;k<microtilesize;k++){
                                    outs[k+tileoffset][j/numthreadsqr]/=newvalue;
                                }
                            }
                            if (microrow!=0){
                                for (int j=i; j<microtilesize; j+=numthreadsqr){
                                    out[(iter+diagstartidx+microdiagoffset+microrowoffset)*size_in+j+diagstartidx+microdiagoffset]=outs[iter+microtilesize][j/numthreadsqr];
                                }
                            }
                            
                        }
                    } else {
                        for (int j=(1+microiter)*microtilesize+i, l=0; j<microtilesize*nummicrotiles; j+=numthreadsqr*(nummicroblocksqr-1), l++){
                            
                            float newvalue=cache2[1] + (cache2[1]>0 ? 1 : -1) * sqrt(cache2[0]+cache2[1]*cache2[1]); //u1
                            float taut=2 /(cache2[0]/(newvalue* newvalue)+1);
                            
                            for (int k=startidxk;k<microtilesize;k++){
                                tmp_sum[l]+=cache[k]*outs[k+tileoffset][l];
                            }
                            
                            float tmp_sum2 = ((tmp_sum[l]/newvalue)+outs[iter][l])*taut;
                            outs[iter][l]-=tmp_sum2;
                
                            for (int k=startidxk;k<microtilesize;k++){
                                outs[k+tileoffset][l]-=cache[k]* tmp_sum2/newvalue;
                            }
                    
                        }
                        if (microrow!=0){
                            for (int j=(1+microiter)*microtilesize+i; j<microtilesize*nummicrotiles; j+=numthreadsqr*(nummicroblocksqr-1)){
                                out[(iter+diagstartidx+microdiagoffset+microrowoffset)*size_in+j+diagstartidx+microdiagoffset]=outs[iter+microtilesize][j/numthreadsqr];
                            }
                            
                        }
                            
                        
                    }
                
                    
                    __syncthreads();
                }
                if (microrow==0){
                    if (i2==0){
                        for (int j=i; j<microtilesize; j+=numthreadsqr){
                            out[(microtilesize-1+diagstartidx)*size_in+j+diagstartidx]=outs[microtilesize-1][j/numthreadsqr];
                        }   
                    }else{
                        for (int j=(1+microiter)*microtilesize+i; j<microtilesize*nummicrotiles; j+=numthreadsqr*(nummicroblocksqr-1)){
                            out[(microtilesize-1+diagstartidx+microdiagoffset)*size_in+j+diagstartidx+microdiagoffset]=outs[microtilesize-1][j/numthreadsqr];
                        }
                    }
                    
                }
                
                
            }
            if (i2==0){
                for (int k=0; k<microtilesize;k++){
                    for (int j=i; j<microtilesize; j+=numthreadsqr){
                        out[(k+diagstartidx+microdiagoffset)*size_in+j+diagstartidx+microdiagoffset]=outs[k][j/numthreadsqr];
                    }
                }  
            }else{
                for (int k=0; k<microtilesize;k++){
                    for (int j=(1+microiter)*microtilesize+i; j<microtilesize*nummicrotiles; j+=numthreadsqr*(nummicroblocksqr-1)){
                        out[(k+diagstartidx+microdiagoffset)*size_in+j+diagstartidx+microdiagoffset]=outs[k][j/numthreadsqr];
                    }
                }
            }
        }
        
            
            
    
    }
    
    __global__ __launch_bounds__(numthreadsqr2) void multi_calcQR_doubletile( //calculates in-place QR of diagonal tile combined with row_idx tile below
        int size_in,
        int diag_iter,
        int row_iter,
        float *tau,
        float *out) {
        int j = threadIdx.x;
        float outs[2*microtilesize];
        __shared__ float cache[microtilesize];
        __shared__ float cache2[2];
        int diagstartidx=diag_iter*microtilesize;
        int iteroffset=row_iter*microtilesize;
    
       
        for (int k=0; k<=j;k++){
            outs[k]=out[(k+diagstartidx)*size_in+j+diagstartidx];
        }
        for (int k=0; k<microtilesize;k++){
            outs[k+microtilesize]=out[(k+diagstartidx+iteroffset)*size_in+j+diagstartidx];
        }
        
        
        for(int iter=0;iter<microtilesize;iter++){
            float tmp_sum=0.0f;
            if (j==iter){
                for (int k=0;k<microtilesize;k++){
                    cache[k]=outs[k+microtilesize];
                    tmp_sum+=outs[k+microtilesize]*outs[k+microtilesize];
                }
                cache2[0]=tmp_sum;
                cache2[1]=outs[iter];
            }
            __syncthreads();
    
            if (j>=iter){
                float newvalue=cache2[1] + (cache2[1]>0 ? 1 : -1) * sqrt(cache2[0]+cache2[1]*cache2[1]); //u1
                float taut=2 /(cache2[0]/(newvalue* newvalue)+1);
                if (j>iter){
                    for (int k=0;k<microtilesize;k++){
                        tmp_sum+=cache[k]*outs[k+microtilesize];
                    }
                }else{
                    tau[(diag_iter)*size_in+row_iter*microtilesize+j]=taut;
                }
    
                float tmp_sum2 = (tmp_sum / newvalue+outs[iter])*taut;
                outs[iter]-=tmp_sum2;
                if (j>iter){
                    for (int k=0;k<microtilesize;k++){
                        outs[k+microtilesize]*=newvalue;
                        outs[k+microtilesize]-=cache[k]* tmp_sum2;
                    }
                }
                for (int k=0;k<microtilesize;k++){
                    outs[k+microtilesize]/=newvalue;
                }
            }            
            
            __syncthreads();
    
        }  
            
        for (int k=0; k<=j;k++){
            out[(k+diagstartidx)*size_in+j+diagstartidx]=outs[k];
        }
        for (int k=0; k<microtilesize;k++){
            out[(k+diagstartidx+iteroffset)*size_in+j+diagstartidx]=outs[k+microtilesize];
        }
        
    
        
    }
            
    
    void launch_tiled_qr(
        int32_t size_i,
        float *a, float *tau) {
            
        if ( size_i%(microtilesize*nummicrotiles) !=0 ){
                throw std::invalid_argument( "Not implemented for this argument size" );
        }
        int nb_blocks= size_i/(microtilesize*nummicrotiles);
        for(int iter=0;iter<nb_blocks-1;iter++){
            multi_calcQR_singletile<<<1,dim3(numthreadsqr, nummicroblocksqr)>>>(size_i,iter,tau,a); 
            multi_applyQt_singletile<<<nb_blocks-1-iter,dim3(tilesize)>>>(size_i,size_i,iter,true,tau,a,a); 
            for (int row=1;row+iter<nb_blocks;row++){
                multi_calcQR_doubletile<<<1,dim3(numthreadsqr2)>>>(size_i,iter,row,tau,a);
                multi_applyQt_doubletile<<<nb_blocks-1-iter,dim3(tilesize)>>>(size_i,size_i,iter,row,true,tau,a,a); 
            }
        }
        multi_calcQR_singletile<<<1,dim3(numthreadsqr)>>>(size_i,nb_blocks-1,tau,a); 
            
        }
    
    /*
        void launch_tiled_qr(
            int32_t size_i,
            float *a, float *tau) {
                
            if ( size_i%tilesize !=0 ){
                    throw std::invalid_argument( "Not implemented for this argument size" );
            }
            cudaStream_t stream1, stream2;
            cudaStreamCreate ( &stream1);
            cudaStreamCreate ( &stream2);
            int nb_blocks= size_i/tilesize;
            base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,0,tau,a); 
            cudaDeviceSynchronize ();
            for(int iter=0;iter<nb_blocks-1;iter++){
                base_calcQR_doubletile<<<1,dim3(tilesize,tilesize),0,stream1>>>(size_i,iter,1,tau,a);
                base_applyQt_singletile<<<nb_blocks-1-iter,dim3(tilesize,numthreads),0,stream2>>>(size_i,size_i,iter,true,tau,a,a); 
                cudaDeviceSynchronize ();
                for (int row=1;row+iter<nb_blocks;row++){
                    if (row+iter<nb_blocks-1){
                        base_calcQR_doubletile<<<1,dim3(tilesize,tilesize),0,stream1>>>(size_i,iter,row+1,tau,a);
                    }else if (iter<nb_blocks-2){
                        base_calcQR_singletile<<<1,dim3(tilesize,tilesize),0,stream1>>>(size_i,iter+1,tau,a); 
                    }
                    base_applyQt_doubletile<<<nb_blocks-1-iter,dim3(tilesize,numthreads),0,stream2>>>(size_i,size_i,iter,row,true,tau,a,a); 
                    cudaDeviceSynchronize ();
                }
            }
            cudaStreamDestroy( stream1);
            cudaStreamDestroy(stream2);
            if (nb_blocks>1){
                base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,nb_blocks-1,tau,a);
            }
            }*/
    
    
        void launch_mult_qt(
            int32_t size_k, int32_t size_i,  int32_t size_j,  
            float *a, float *tau, float *b) {
    
            if ( size_k%tilesize !=0 ){
                    throw std::invalid_argument( "Not implemented for this argument size" );
            }
            int no_blocks=size_k/tilesize;
            
            for(int iter=0;iter<min(size_i,size_j)/tilesize;iter++){
                multi_applyQt_singletile<<<no_blocks,dim3(tilesize)>>>(size_j,size_k,iter,false,tau,a, b); 
                for (int row=1;row+iter<(size_i/tilesize);row++){
    
                    multi_applyQt_doubletile<<<no_blocks,dim3(tilesize)>>>(size_j,size_k,iter,row,false,tau,a, b); 
                }
            }
                
            }
    
        void test_qrkernel_single(
            int32_t size_i,
            float *a, float *tau) {
            multi_calcQR_singletile<<<1,dim3(numthreadsqr, nummicroblocksqr)>>>(size_i,0,tau,a); 
        
        
            }
    
        void test_mulqtkernel_single(
            int32_t size_i,
            float *a, float *tau) {
                multi_calcQR_singletile<<<1,dim3(numthreadsqr)>>>(size_i,0,tau,a); 
                multi_applyQt_singletile<<<1,dim3(tilesize)>>>(size_i,size_i,0,true,tau,a,a); 
    
    
        
            }
        void test_qrkernel_double(
            int32_t size_i,
            float *a, float *tau) {
            multi_calcQR_singletile<<<1,dim3(numthreadsqr)>>>(size_i,0,tau,a); 
            multi_calcQR_doubletile<<<1,dim3(numthreadsqr)>>>(size_i,0,1,tau,a); 
        
        
            }
    
        void test_mulqtkernel_double(
            int32_t size_i,
            float *a, float *tau) {
                multi_calcQR_singletile<<<1,dim3(numthreadsqr)>>>(size_i,0,tau,a); 
                multi_applyQt_singletile<<<1,dim3(tilesize)>>>(size_i,size_i,0,true,tau,a,a); 
                multi_calcQR_doubletile<<<1,dim3(numthreadsqr)>>>(size_i,0,1,tau,a);
                multi_applyQt_doubletile<<<1,dim3(tilesize)>>>(size_i,size_i,0,1,true,tau,a,a); 
                multi_calcQR_singletile<<<1,dim3(numthreadsqr)>>>(size_i,1,tau,a); 
        
            }         
                    
            
    };

/*

namespace qr_base {

#define tilesize 4
#define numthreads 1
#define numilpqr 1
#define numthreadsqr 4
#define numthreadsqr2 4


//-------------------- multiply by Q kernels-----------------------------------


    
    __global__ void multi_applyQt_singletile( //aplies Qt (given by householder reflectors on diagonal tile k) to the remainder of the row
        int size_in,
        int size_out,
        int diag_iter,
        bool offsetdiag,
        float const *tau,
        float *in, float *out) {
        int g = blockIdx.x;
        int i = threadIdx.x;
        float outs[tilesize];
        __shared__ float Qs[tilesize];
        int diagstartidx=diag_iter*tilesize;
        int tileoffset=(g)*tilesize;
        if (offsetdiag){
            tileoffset+=diagstartidx+tilesize;
        }
        
        
        for (int l=0;l<tilesize;l++){
            outs[l]=out[(l+diagstartidx)*size_out+i+tileoffset];
        }
        
    
        __syncthreads();
    
        for (int k=0;k<tilesize-1;k++){
            Qs[i]=in[(i+diagstartidx)*size_in+k+diagstartidx];
            __syncthreads();
            float tmp_sum = 0.0f;
            for (int l=k+1;l<tilesize;l++){
                tmp_sum+= Qs[l]*outs[l];
            }
            tmp_sum+=outs[k];
            tmp_sum*=tau[(diag_iter)*size_in+k];
            for (int l=k+1;l<tilesize;l++){
                outs[l]-=tmp_sum*Qs[l];
            }
            outs[k]-=tmp_sum;
            __syncthreads();
        }
    
        for (int l=0;l<tilesize;l++){
            out[(l+diagstartidx)*size_out+i+tileoffset]=outs[l];
        }
    
    }
    
    __global__ void multi_applyQt_doubletile( //aplies Qt (given by householder reflectors on the tile at row_idx below diag_idx) to the remainder of the row, and to the row of diag_idx
        int size_in,
        int size_out,
        int diag_iter,
        int row_iter,
        bool offsetdiag,
        float const *tau,
        float *in, float *out) {
        int g = blockIdx.x;
        int i = threadIdx.x;
        float outs[2*tilesize];
        __shared__ float Qs[tilesize];
        int diagstartidx=diag_iter*tilesize;
        int tileoffset=(g)*tilesize;
        int iteroffset=row_iter*tilesize;
        if (offsetdiag){
            tileoffset+=diagstartidx+tilesize;
        }
    
        for (int l=0;l<tilesize;l++){
            outs[l]=out[(l+diagstartidx)*size_out+i+tileoffset];
            outs[l+tilesize]=out[(l+diagstartidx+iteroffset)*size_out+i+tileoffset];
        }
    
        __syncthreads();
    
    
        for (int k=0;k<tilesize;k++){
            Qs[i]=in[(i+diagstartidx+iteroffset)*size_in+k+diagstartidx];
            __syncthreads();
            float tmp_sum = 0.0f;
            for (int l=0;l<tilesize;l++){
                tmp_sum+= Qs[l]*outs[l+tilesize];
            }
            tmp_sum+=outs[k] ;
            tmp_sum*=tau[(diag_iter)*size_in+row_iter*tilesize+k];
            outs[k]-=tmp_sum;
            for (int l=0;l<tilesize;l++){
                outs[l+tilesize]-=tmp_sum*Qs[l];
            }
            
            __syncthreads();
        }
    
        for (int l=0;l<tilesize;l++){
            out[(l+diagstartidx)*size_out+i+tileoffset]=outs[l];
            out[(l+diagstartidx+iteroffset)*size_out+i+tileoffset]=outs[l+tilesize];
        }
    
    }
//-------------------- calculate QR kernels-----------------------------------



__global__ __launch_bounds__(numthreadsqr) void base_calcQR_singletile( //calculates in-place QR of diagonal tile
    int size_in,
    int diag_iter,
    float *tau,
    float *out) {
        
    int i = threadIdx.x;
    float outs[tilesize][numilpqr];
    float taureg[numilpqr];
    __shared__ float cache[tilesize];
    __shared__ float tauvals[2];
    int diagstartidx=diag_iter*tilesize;
        
    for (int k=0; k<tilesize;k++){
        for (int j=i; j<tilesize; j+=numthreadsqr){
            outs[k][j/numthreadsqr]=out[(k+diagstartidx)*size_in+j+diagstartidx];
        }
    }
    
    __syncthreads();

    for(int iter=0;iter<tilesize-1;iter++){
        if (i==(iter%numthreadsqr)){
            for (int k=iter+1;k<tilesize;k++){
                cache[k]=outs[k][iter/numthreadsqr];
            }
        }
        __syncthreads();
        float tmp_sum[numilpqr]={0.0f};
        
        for (int j=i; j<tilesize; j+=numthreadsqr){
            if (j>=iter){
                for (int k=iter+1;k<tilesize;k++){
                    tmp_sum[j/numthreadsqr]+=cache[k]*outs[k][j/numthreadsqr];
                }
                if (j==iter){
                   float tmp_sum2=sqrt(tmp_sum[j/numthreadsqr]+pow(outs[iter][iter/numthreadsqr],2)); //normx
                    float newvalue=outs[iter][iter/numthreadsqr] + (outs[iter][iter/numthreadsqr]>0 ? 1 : -1) * tmp_sum2; //u1
                    tmp_sum2=sqrt(tmp_sum[j/numthreadsqr]+pow(newvalue,2));
                    taureg[iter/numthreadsqr]=2 * pow(newvalue/tmp_sum2,2);
                    tauvals[1]= newvalue;
                    tauvals[0]=taureg[iter/numthreadsqr]; 
                }
            }
        }
        __syncthreads();
        for (int j=i; j<tilesize; j+=numthreadsqr){
            if (j>iter){
                float tmp_sum2 = (tmp_sum[j/numthreadsqr] / tauvals[1]+outs[iter][j/numthreadsqr])*tauvals[0];
                for (int k=iter+1;k<tilesize;k++){
                    outs[k][j/numthreadsqr]-=cache[k]* tmp_sum2/ tauvals[1];
                }
                outs[iter][j/numthreadsqr]-=tmp_sum2;
            }else if (j==iter){
                float tmp_sum2 = (tmp_sum[j/numthreadsqr] / tauvals[1]+outs[iter][j/numthreadsqr])*tauvals[0];
                for (int k=iter+1;k<tilesize;k++){
                    outs[k][j/numthreadsqr]/=tauvals[1];
                }
                outs[iter][j/numthreadsqr]-=tmp_sum2;
            }
            out[(iter+diagstartidx)*size_in+j+diagstartidx]=outs[iter][j/numthreadsqr];
        }


        
    }
    __syncthreads();
    for (int j=i; j<tilesize; j+=numthreadsqr){
        out[(tilesize-1+diagstartidx)*size_in+j+diagstartidx]=outs[tilesize-1][j/numthreadsqr];
        tau[(diag_iter)*size_in+j]=taureg[j/numthreadsqr];
    }
        
        

}

__global__ __launch_bounds__(numthreadsqr2) void base_calcQR_doubletile( //calculates in-place QR of diagonal tile combined with row_idx tile below
    int size_in,
    int diag_iter,
    int row_iter,
    float *tau,
    float *out) {
    int j = threadIdx.x;
    
    float outs[2*tilesize];
    float taureg;
    __shared__ float cache[tilesize+2];
    int diagstartidx=diag_iter*tilesize;
    int iteroffset=row_iter*tilesize;

   
        for (int k=0; k<=j;k++){
            outs[k]=out[(k+diagstartidx)*size_in+j+diagstartidx];
        }
        for (int k=0; k<tilesize;k++){
            outs[k+tilesize]=out[(k+diagstartidx+iteroffset)*size_in+j+diagstartidx];
        }
    
    
    for(int iter=0;iter<tilesize;iter++){
        float tmp_sum=0.0f;
        if (j==iter){
            for (int k=0;k<tilesize;k++){
                cache[k]=outs[k+tilesize];
                tmp_sum+=outs[k+tilesize]*outs[k+tilesize];
            }
            cache[tilesize]=tmp_sum;
            cache[tilesize+1]=outs[iter];
        }
        __syncthreads();

        if (j>=iter){
            float newvalue=cache[tilesize+1] + (cache[tilesize+1]>0 ? 1 : -1) * sqrt(cache[tilesize]+cache[tilesize+1]*cache[tilesize+1]); //u1
            float taut=2 /(cache[tilesize]/(newvalue* newvalue)+1);
            if (j>iter){
                for (int k=0;k<tilesize;k++){
                    tmp_sum+=cache[k]*outs[k+tilesize];
                }
            }else{
                taureg=taut;
            }

            float tmp_sum2 = (tmp_sum / newvalue+outs[iter])*taut;
            outs[iter]-=tmp_sum2;
            if (j>iter){
                for (int k=0;k<tilesize;k++){
                    outs[k+tilesize]*=newvalue;
                    outs[k+tilesize]-=cache[k]* tmp_sum2;
                }
            }
            for (int k=0;k<tilesize;k++){
                outs[k+tilesize]/=newvalue;
            }
        }            
        
        __syncthreads();

    }

        
        for (int k=0; k<=j;k++){
            out[(k+diagstartidx)*size_in+j+diagstartidx]=outs[k];
        }
        for (int k=0; k<tilesize;k++){
            out[(k+diagstartidx+iteroffset)*size_in+j+diagstartidx]=outs[k+tilesize];
        }
        tau[(diag_iter)*size_in+row_iter*tilesize+j]=taureg;

    
}
        

void launch_tiled_qr(
    int32_t size_i,
    float *a, float *tau) {
        
    if ( size_i%tilesize !=0 ){
            throw std::invalid_argument( "Not implemented for this argument size" );
    }
    int nb_blocks= size_i/tilesize;
    for(int iter=0;iter<nb_blocks-1;iter++){
        base_calcQR_singletile<<<1,dim3(numthreadsqr)>>>(size_i,iter,tau,a); 
        base_applyQt_singletile<<<nb_blocks-1-iter,dim3(tilesize,numthreads)>>>(size_i,size_i,iter,true,tau,a,a); 
        for (int row=1;row+iter<nb_blocks;row++){
            base_calcQR_doubletile<<<1,dim3(numthreadsqr2)>>>(size_i,iter,row,tau,a);
            base_applyQt_doubletile<<<nb_blocks-1-iter,dim3(tilesize,numthreads)>>>(size_i,size_i,iter,row,true,tau,a,a); 
        }
    }
    base_calcQR_singletile<<<1,dim3(numthreadsqr)>>>(size_i,nb_blocks-1,tau,a); 
        
    }

/*
    void launch_tiled_qr(
        int32_t size_i,
        float *a, float *tau) {
            
        if ( size_i%tilesize !=0 ){
                throw std::invalid_argument( "Not implemented for this argument size" );
        }
        cudaStream_t stream1, stream2;
        cudaStreamCreate ( &stream1);
        cudaStreamCreate ( &stream2);
        int nb_blocks= size_i/tilesize;
        base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,0,tau,a); 
        cudaDeviceSynchronize ();
        for(int iter=0;iter<nb_blocks-1;iter++){
            base_calcQR_doubletile<<<1,dim3(tilesize,tilesize),0,stream1>>>(size_i,iter,1,tau,a);
            base_applyQt_singletile<<<nb_blocks-1-iter,dim3(tilesize,numthreads),0,stream2>>>(size_i,size_i,iter,true,tau,a,a); 
            cudaDeviceSynchronize ();
            for (int row=1;row+iter<nb_blocks;row++){
                if (row+iter<nb_blocks-1){
                    base_calcQR_doubletile<<<1,dim3(tilesize,tilesize),0,stream1>>>(size_i,iter,row+1,tau,a);
                }else if (iter<nb_blocks-2){
                    base_calcQR_singletile<<<1,dim3(tilesize,tilesize),0,stream1>>>(size_i,iter+1,tau,a); 
                }
                base_applyQt_doubletile<<<nb_blocks-1-iter,dim3(tilesize,numthreads),0,stream2>>>(size_i,size_i,iter,row,true,tau,a,a); 
                cudaDeviceSynchronize ();
            }
        }
        cudaStreamDestroy( stream1);
        cudaStreamDestroy(stream2);
        if (nb_blocks>1){
            base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,nb_blocks-1,tau,a);
        }
        }*//*


    void launch_mult_qt(
        int32_t size_k, int32_t size_i,  int32_t size_j,  
        float *a, float *tau, float *b) {

        if ( size_k%tilesize !=0 ){
                throw std::invalid_argument( "Not implemented for this argument size" );
        }
        int no_blocks=size_k/tilesize;
        
        for(int iter=0;iter<min(size_i,size_j)/tilesize;iter++){
            base_applyQt_singletile<<<no_blocks,dim3(tilesize,numthreads)>>>(size_j,size_k,iter,false,tau,a, b); 
            for (int row=1;row+iter<(size_i/tilesize);row++){

                base_applyQt_doubletile<<<no_blocks,dim3(tilesize,numthreads)>>>(size_j,size_k,iter,row,false,tau,a, b); 
            }
        }
            
        }

    void test_qrkernel_single(
        int32_t size_i,
        float *a, float *tau) {
        base_calcQR_singletile<<<1,dim3(numthreadsqr)>>>(size_i,0,tau,a); 
    
    
        }

    void test_mulqtkernel_single(
        int32_t size_i,
        float *a, float *tau) {
            base_calcQR_singletile<<<1,dim3(numthreadsqr)>>>(size_i,0,tau,a); 
            base_applyQt_singletile<<<1,dim3(tilesize,numthreads)>>>(size_i,size_i,0,true,tau,a,a); 


    
        }
    void test_qrkernel_double(
        int32_t size_i,
        float *a, float *tau) {
        base_calcQR_singletile<<<1,dim3(numthreadsqr)>>>(size_i,0,tau,a); 
        base_calcQR_doubletile<<<1,dim3(numthreadsqr)>>>(size_i,0,1,tau,a); 
    
    
        }

    void test_mulqtkernel_double(
        int32_t size_i,
        float *a, float *tau) {
            base_calcQR_singletile<<<1,dim3(numthreadsqr)>>>(size_i,0,tau,a); 
            base_applyQt_singletile<<<1,dim3(tilesize,numthreads)>>>(size_i,size_i,0,true,tau,a,a); 
            base_calcQR_doubletile<<<1,dim3(numthreadsqr)>>>(size_i,0,1,tau,a);
            base_applyQt_doubletile<<<1,dim3(tilesize,numthreads)>>>(size_i,size_i,0,1,true,tau,a,a); 
            base_calcQR_singletile<<<1,dim3(numthreadsqr)>>>(size_i,1,tau,a); 
    
        }         
                
        
};

////////////////////////////////////////////////////////////////////////////////

namespace qr_base_improved {
    
    
    //-------------------- multiply by Q kernels-----------------------------------
    
    
    
  
__global__ void base_applyQt_singletile( //aplies Qt (given by householder reflectors on diagonal tile k) to the remainder of the row
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

__global__ void base_applyQt_doubletile( //aplies Qt (given by householder reflectors on the tile at row_idx below diag_idx) to the remainder of the row, and to the row of diag_idx
    int size_in,
    int size_out,
    int diag_iter,
    int row_iter,
    bool offsetdiag,
    float const *tau,
    float *in, float *out) {
    int g = blockIdx.x;
    int i = threadIdx.x;
    int j = threadIdx.y;
    __shared__ float outs[2*tilesize][tilesize];
    __shared__ float Qs[tilesize][tilesize];
    __shared__ float cache[tilesize][numthreads];
    int diagstartidx=diag_iter*tilesize;
    int tileoffset=(g)*tilesize;
    int iteroffset=row_iter*tilesize;
    if (offsetdiag){
        tileoffset+=diagstartidx+tilesize;
    }
    
    
    for (int l=j;l<tilesize;l+=numthreads){
        outs[i][l]=out[(i+diagstartidx)*size_out+l+tileoffset];
        outs[i+tilesize][l]=out[(i+diagstartidx+iteroffset)*size_out+l+tileoffset];
        Qs[i][l]=in[(i+diagstartidx+iteroffset)*size_in+l+diagstartidx];
    }

    __syncthreads();


    for (int k=0;k<tilesize;k++){
        float tmp_sum = 0.0f;
        for (int l=j;l<tilesize;l+=numthreads){
            tmp_sum+= Qs[l][k]*outs[l+tilesize][i];
        }
        cache[i][j]=tmp_sum;
        __syncthreads();
        tmp_sum=outs[k][i];
        for (int l=0;l<numthreads;l++){
            tmp_sum+=cache[i][l];
        }
        tmp_sum*=tau[(diag_iter)*size_in+row_iter*tilesize+k];
        if (j==0){
            outs[k][i]-=tmp_sum;
        }
        for (int l=j;l<tilesize;l+=numthreads){
            outs[l+tilesize][i]-=tmp_sum*Qs[l][k];
        }
        
        __syncthreads();
    }

    for (int l=j;l<tilesize;l+=numthreads){
        out[(i+diagstartidx)*size_out+l+tileoffset]=outs[i][l];
        out[(i+diagstartidx+iteroffset)*size_out+l+tileoffset]=outs[i+tilesize][l];
    }

}
//-------------------- calculate QR kernels-----------------------------------



__global__ void base_calcQR_singletile( //calculates in-place QR of diagonal tile
    int size_in,
    int diag_iter,
    float *tau,
    float *out) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    __shared__ float outs[tilesize][tilesize];
    __shared__ float cache[tilesize];
    __shared__ float tauvals[2];
    int diagstartidx=diag_iter*tilesize;

    outs[i][j]=out[(i+diagstartidx)*size_in+j+diagstartidx];

    __syncthreads();

    for(int iter=0;iter<tilesize-1;iter++){
        if (i>iter && j==0){
            cache[i]=outs[i][iter]*outs[i][iter];
        } 
        
        __syncthreads();
        if (i==0 && j==0){
            float tmp_sum=0.0f;
            for (int l=iter+1;l<tilesize;l++){
                tmp_sum+=cache[l];
             }
            float tmp_sum2=sqrt(tmp_sum+pow(outs[iter][iter],2));
            float newvalue=outs[iter][iter];
            if (newvalue>0){
                newvalue+=tmp_sum2;
            }else{
                newvalue-=tmp_sum2;
            }
            tmp_sum2=sqrt(tmp_sum+pow(newvalue,2));
            tauvals[0]=2 * pow(newvalue/tmp_sum2,2);
            tauvals[1]= newvalue;
            tau[(diag_iter)*size_in+iter]=tauvals[0];

        }
        float tmp_sum=0.0f;
        if (j>=iter && i>=iter){
            for (int k=iter+1;k<tilesize;k++){
                tmp_sum+=outs[k][iter]*outs[k][j];
            }
        }
        float tileiterj=outs[iter][j];
        float tileiiter = outs[i][iter];
        __syncthreads();
        if (j>=iter && i>=iter){
            tmp_sum = (tmp_sum / tauvals[1]+tileiterj)*tauvals[0];
            
            
            tileiiter/=tauvals[1];
            
            if (j==iter && i>iter){
                outs[i][j]=tileiiter;
            }else if(i>iter){
                outs[i][j]-=tileiiter*tmp_sum;
                
            }else if (i==iter){
                outs[i][j]-=tmp_sum;
            }
            
        }
        __syncthreads();
        out[(i+diagstartidx)*size_in+j+diagstartidx]=outs[i][j];
    }

}

__global__ void base_calcQR_doubletile( //calculates in-place QR of diagonal tile combined with row_idx tile below
    int size_in,
    int diag_iter,
    int row_iter,
    float *tau,
    float *out) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    __shared__ float outs[2*tilesize][tilesize];
    __shared__ float cache[2*tilesize];
    __shared__ float tauvals[2];
    int diagstartidx=diag_iter*tilesize;
    int iteroffset=row_iter*tilesize;

    outs[i][j]=out[(i+diagstartidx)*size_in+j+diagstartidx];
    outs[i+tilesize][j]=out[(i+diagstartidx+iteroffset)*size_in+j+diagstartidx];
    
    for(int iter=0;iter<tilesize;iter++){
        if (j==iter){
            cache[i]=outs[i+tilesize][iter]*outs[i+tilesize][iter];
        }
        __syncthreads();
        if (i==0 && j==0){
            float tmp_sum=0.0f;
            for (int l=0;l<tilesize;l++){
                tmp_sum+=cache[l];
            }
            float tmp_sum2=sqrt(tmp_sum+outs[iter][iter]*outs[iter][iter]);
            float newvalue=outs[iter][iter];
            if (newvalue>0){
                newvalue+=tmp_sum2;
            }else{
                newvalue-=tmp_sum2;
            }
            tmp_sum2=sqrt(tmp_sum+newvalue*newvalue);
            tauvals[0]=2 * (newvalue/tmp_sum2)*(newvalue/tmp_sum2);
            tauvals[1]= newvalue;
            tau[(diag_iter)*size_in+row_iter*tilesize+iter]=tauvals[0];
        }
        float tileiterj=outs[iter][j];
        float tileiiter = outs[i+tilesize][iter];
        float tmp_sum=0.0f;
        if (j>=iter){
            for (int k=tilesize;k<tilesize*2;k++){
                tmp_sum+=outs[k][iter]*outs[k][j];
            }
        }

        __syncthreads();
        if (j>=iter){ 
            tmp_sum = ( tmp_sum/tauvals[1]+tileiterj)*tauvals[0];
            if (i==iter){ //j>=iter && k==iter
                outs[i][j]-=tmp_sum;
            }
            if(j>iter){// k>iter && j>iter
                outs[i+tilesize][j]-=tileiiter*tmp_sum /tauvals[1];
            }
            
    

        }
        __syncthreads();
        if (j==0){ 
            outs[i+tilesize][iter]=tileiiter / tauvals[1];
        }
        __syncthreads();

    }
    __syncthreads();
    out[(i+diagstartidx)*size_in+j+diagstartidx]=outs[i][j];
    out[(i+diagstartidx+iteroffset)*size_in+j+diagstartidx]=outs[i+tilesize][j];
        
    
}

    
    
void launch_tiled_qr(
    int32_t size_i,
    float *a, float *tau) {
        
    if ( size_i%tilesize !=0 ){
            throw std::invalid_argument( "Not implemented for this argument size" );
    }
    int nb_blocks= size_i/tilesize;
    for(int iter=0;iter<nb_blocks-1;iter++){
        base_calcQR_singletile<<<1,dim3(tilesize, tilesize)>>>(size_i,iter,tau,a); 
        base_applyQt_singletile<<<nb_blocks-1-iter,dim3(tilesize,numthreads)>>>(size_i,size_i,iter,true,tau,a,a); 
        for (int row=1;row+iter<nb_blocks;row++){
            base_calcQR_doubletile<<<1,dim3(tilesize,tilesize)>>>(size_i,iter,row,tau,a);
            base_applyQt_doubletile<<<nb_blocks-1-iter,dim3(tilesize,numthreads)>>>(size_i,size_i,iter,row,true,tau,a,a); 
        }
    }
    base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,nb_blocks-1,tau,a); 
        
    }/*
    
        void launch_tiled_qr(
            int32_t size_i,
            float *a, float *tau) {
                
            if ( size_i%tilesize !=0 ){
                    throw std::invalid_argument( "Not implemented for this argument size" );
            }
            cudaStream_t stream1, stream2;
            cudaStreamCreate ( &stream1);
            cudaStreamCreate ( &stream2);
            int nb_blocks= size_i/tilesize;
            base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,0,tau,a); 
            cudaDeviceSynchronize ();
            for(int iter=0;iter<nb_blocks-1;iter++){
                base_calcQR_doubletile<<<1,dim3(tilesize,tilesize),0,stream1>>>(size_i,iter,1,tau,a);
                base_applyQt_singletile<<<nb_blocks-1-iter,dim3(tilesize,numthreads),0,stream2>>>(size_i,size_i,iter,true,tau,a,a); 
                cudaDeviceSynchronize ();
                for (int row=1;row+iter<nb_blocks;row++){
                    if (row+iter<nb_blocks-1){
                        base_calcQR_doubletile<<<1,dim3(tilesize,tilesize),0,stream1>>>(size_i,iter,row+1,tau,a);
                    }else if (iter<nb_blocks-2){
                        base_calcQR_singletile<<<1,dim3(tilesize,tilesize),0,stream1>>>(size_i,iter+1,tau,a); 
                    }
                    base_applyQt_doubletile<<<nb_blocks-1-iter,dim3(tilesize,numthreads),0,stream2>>>(size_i,size_i,iter,row,true,tau,a,a); 
                    cudaDeviceSynchronize ();
                }
            }
            cudaStreamDestroy( stream1);
            cudaStreamDestroy(stream2);
            if (nb_blocks>1){
                base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,nb_blocks-1,tau,a);
            }
            }*//*
    
    
        void launch_mult_qt(
            int32_t size_k, int32_t size_i,  int32_t size_j,  
            float *a, float *tau, float *b) {
    
            if ( size_k%tilesize !=0 ){
                    throw std::invalid_argument( "Not implemented for this argument size" );
            }
            int no_blocks=size_k/tilesize;
            
            for(int iter=0;iter<min(size_i,size_j)/tilesize;iter++){
                base_applyQt_singletile<<<no_blocks,dim3(tilesize,numthreads)>>>(size_j,size_k,iter,false,tau,a, b); 
                for (int row=1;row+iter<(size_i/tilesize);row++){
    
                    base_applyQt_doubletile<<<no_blocks,dim3(tilesize,numthreads)>>>(size_j,size_k,iter,row,false,tau,a, b); 
                }
            }
                
            }
    
        void test_qrkernel_single(
            int32_t size_i,
            float *a, float *tau) {
            base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,0,tau,a); 
        
        
            }
    
        void test_mulqtkernel_single(
            int32_t size_i,
            float *a, float *tau) {
                base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,0,tau,a); 
                base_applyQt_singletile<<<1,dim3(tilesize,numthreads)>>>(size_i,size_i,0,true,tau,a,a); 
    
    
        
            }
        void test_qrkernel_double(
            int32_t size_i,
            float *a, float *tau) {
            base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,0,tau,a); 
            base_calcQR_doubletile<<<1,dim3(tilesize,tilesize)>>>(size_i,0,1,tau,a); 
        
        
            }
    
        void test_mulqtkernel_double(
            int32_t size_i,
            float *a, float *tau) {
                base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,0,tau,a); 
                base_applyQt_singletile<<<1,dim3(tilesize,numthreads)>>>(size_i,size_i,0,true,tau,a,a); 
                base_calcQR_doubletile<<<1,dim3(tilesize,tilesize)>>>(size_i,0,1,tau,a);
                base_applyQt_doubletile<<<1,dim3(tilesize,numthreads)>>>(size_i,size_i,0,1,true,tau,a,a); 
                base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,1,tau,a); 
        
            }         
                    
            
    
    
    };
*/

////////////////////////////////////////////////////////////////////////////////

enum class Phase {
    TEST,
    WARMUP,
    BENCHMARK,
};

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t, int32_t>, double> elapsed_ms;
    std::map<std::tuple<int32_t,int32_t>, double> elapsed_ms_mulx;
};


struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
};

struct TestData {
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> ref;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> x;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> xref;
};

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
benchmark_ms( Reset &&reset, F &&f) {
    double target_time_ms = 200.0;
    int32_t num_iters_inner =4;
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

float get_relrmse(int size_i, int size_j, float *valsgpu, std::vector<float> ref, bool triu, Phase phase ){
    std::vector<float> out_host(size_i * size_j * sizeof(float));
    CUDA_CHECK(cudaMemcpy(  out_host.data(), valsgpu ,size_i * size_j* sizeof(float), cudaMemcpyDeviceToHost));

    double mse = 0.0;
    double divmse=0.0;
    for (int32_t i = 0; i < min(size_i,size_j); ++i) {
        for (int32_t j = (triu? i : 0); j < size_j; ++j) {
            float diff = abs(abs(out_host[i * size_j + j]) - abs(ref[i * size_j + j]));
            mse+= diff*diff;
            divmse+= abs(ref[i * size_j + j])*abs(ref[i * size_j + j]);
        }
    }

    float rel_rmse = std::sqrt(mse) / std::sqrt(divmse) ;

    if (((rel_rmse > 1e-3)  || rel_rmse!=rel_rmse) && size_i<65 && size_j<65 && phase == Phase::TEST) {
        printf("\n  expected output:\n");
        print_matrix(size_i, size_j, ref);
        printf("\n  obtained output:\n");
        print_matrix(size_i,  size_j, out_host);
    } 
    return rel_rmse;
}


TestData read_test_data(Phase phase,
    std::string const &test_data_dir,
    std::vector<BenchmarkConfig> const &configs) {
    auto data = TestData{};
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto path_prefix = test_data_dir + "/test_";

        if (data.a.find({size_i, size_j}) == data.a.end()) {
            data.a[{size_i, size_j}] = read_data(
                path_prefix + "a_" + std::to_string(size_i) + "_" +
                    std::to_string(size_j) + ".bin",
                size_i * size_j);
        }

        if (data.ref.find({size_i, size_j}) == data.ref.end()) {
            data.ref[{size_i, size_j}] = read_data(
                path_prefix + "a_" + std::to_string(size_i) + "_" +
                    std::to_string(size_j) + ".bin",
                size_i * size_j);
        }
        if (data.x.find({size_i, size_j}) == data.x.end()) {
            data.x[{size_i, size_j}] = read_data(
                path_prefix + "x_" + std::to_string(size_i) +"_" +
                std::to_string(size_j) + ".bin",
                size_i * size_j);
        }
        

        if (data.xref.find({size_i, size_j}) == data.xref.end()) {
            data.xref[{size_i, size_j}] = read_data(
                path_prefix + "x_" + std::to_string(size_i) +"_" +
                std::to_string(size_j) + ".bin",
                size_i * size_j);
        }
        
            

    }
    return data;
}


template <typename Impl>
void run_config(
    Phase phase,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size_i = config.size_i;
    auto size_j = config.size_j;

    auto const &a = data.a.at({size_i, size_j});
    auto const &ref = data.ref.at({size_i, size_j});

    float *a_gpu;
    float *tau_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_j * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tau_gpu, size_i * size_j * sizeof(float))); // TODO: determine size more accurately

    CUDA_CHECK(cudaMemcpy(   a_gpu,a.data(),size_i * size_j * sizeof(float),cudaMemcpyHostToDevice));


    if (phase == Phase::TEST){
        if(size_i==tilesize && size_j==tilesize){
            Impl::testqr(size_j, a_gpu, tau_gpu);
        }else if (size_i==tilesize){
            Impl::testmulq(size_j, a_gpu, tau_gpu);
        }else if (size_j==tilesize){
            Impl::testqr2(size_j, a_gpu, tau_gpu);
        }else if (size_i==2*tilesize && size_j==2*tilesize){
            Impl::testmulq2(size_j, a_gpu, tau_gpu);
        } else{
            Impl::run(size_j,   a_gpu, tau_gpu); 
        }
    }
    else{
        Impl::run(size_j,   a_gpu, tau_gpu);
    }

    float rel_rmse = get_relrmse(size_i, size_j, a_gpu, ref, true, phase );
    
    
    if (phase == Phase::BENCHMARK){
        double elapsed_ms = benchmark_ms(
            [&]() {
                CUDA_CHECK(cudaMemcpy( a_gpu, a.data(), size_i * size_j * sizeof(float),cudaMemcpyHostToDevice));
            },
            [&]() {
                Impl::run(size_j,  a_gpu, tau_gpu);
            });
            printf(" %6d  %6d   QR     ", size_i, size_j);
            printf(" %8.02e ", rel_rmse);
            printf(" %9.02f ", elapsed_ms);
            results.elapsed_ms[{size_i, size_j}] = elapsed_ms;
            CUDA_CHECK(cudaMemcpy( a_gpu, a.data(), size_i * size_j * sizeof(float),cudaMemcpyHostToDevice));
            Impl::run(size_j,   a_gpu, tau_gpu);
    } 

    
    auto const &x = data.x.at({size_i, size_j});
    auto const &xref = data.xref.at({size_i, size_j});

    float *x_gpu;
    CUDA_CHECK(cudaMalloc(&x_gpu, size_i * size_j * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(   x_gpu, x.data(),size_i * size_j * sizeof(float),cudaMemcpyHostToDevice));

    Impl::run_mulqt(size_j, size_i,size_j, a_gpu, tau_gpu, x_gpu);
    rel_rmse = get_relrmse(size_i, size_j, x_gpu, xref, false, phase );

    
    if (phase == Phase::BENCHMARK){
        double elapsed_ms = benchmark_ms(
            [&]() {
                CUDA_CHECK(cudaMemcpy( x_gpu, x.data(), size_i * size_j * sizeof(float),cudaMemcpyHostToDevice));
            },
            [&]() {
                Impl::run_mulqt(size_j, size_i,size_j, a_gpu, tau_gpu, x_gpu);
            });
            printf("  Qmul      %8.02e", rel_rmse);
            printf(" %9.02f \n", elapsed_ms);
            results.elapsed_ms_mulx[{size_i,size_j}] = elapsed_ms;    
    }

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(tau_gpu));
    CUDA_CHECK(cudaFree(x_gpu));
}

template <typename Impl>
BenchmarkResults run_all_configs(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{Impl::name};
    if (phase == Phase::WARMUP) {
    } else if (phase == Phase::TEST){
        printf(" testing %s", Impl::name);
    }else {
        printf("%s:\n\n", Impl::name);
        printf(
            " %-6s  %-6s  %-6s     %-8s%-9s%-6s     %-8s  %-9s  \n",
            "size_i",
            "size_j",
            " type ",
            "RRMSE ",
            "time (ms)",
            " type ",
            "RRMSE ",
            "time (ms)");
        printf(
            " %-6s  %-6s  %-6s    %-8s %-9s  %-6s    %-8s %-9s\n",
            "------",
            "------",
            "------",
            "--------",
            "---------",
            "------",
            "--------",
            "---------");
    }
    for (auto const &config : configs) {
        run_config<Impl>(phase, data, config, results);
    }
    printf("\n");
    return results;
}


struct QRmulti {
    constexpr static char const *name = "qr_multi";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j) {
        return 0;
    }

    static void
    run(int32_t size_i,
        float *a, float *tau) {
        qr_multi::launch_tiled_qr(size_i,a, tau) ;
    }

    static void
    run_mulqt(int32_t size_k,int32_t size_i, int32_t size_j, 
        float *a, float *tau, float *b) {
        qr_multi::launch_mult_qt(size_k,size_i,size_j,a, tau, b) ;
    }


    static void
    testqr(int32_t size_i,
        float *a, float *tau) {
        qr_multi::test_qrkernel_single(size_i,a, tau) ;
    }

    static void
    testqr2(int32_t size_i,
        float *a, float *tau) {
        qr_multi::test_qrkernel_double(size_i,a, tau) ;
    }
    static void
    testmulq(int32_t size_i,
        float *a, float *tau) {
        qr_multi::test_mulqtkernel_single(size_i,a, tau) ;
    }

    static void
    testmulq2(int32_t size_i,
        float *a, float *tau) {
        qr_multi::test_mulqtkernel_double(size_i,a, tau) ;
    }


};
/*
struct QRbase {
    constexpr static char const *name = "qr_base";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j) {
        return 0;
    }

    static void
    run(int32_t size_i,
        float *a, float *tau) {
        qr_base::launch_tiled_qr(size_i,a, tau) ;
    }

    static void
    run_mulqt(int32_t size_k,int32_t size_i, int32_t size_j, 
        float *a, float *tau, float *b) {
        qr_base::launch_mult_qt(size_k,size_i,size_j,a, tau, b) ;
    }


    static void
    testqr(int32_t size_i,
        float *a, float *tau) {
        qr_base::test_qrkernel_single(size_i,a, tau) ;
    }

    static void
    testqr2(int32_t size_i,
        float *a, float *tau) {
        qr_base::test_qrkernel_double(size_i,a, tau) ;
    }
    static void
    testmulq(int32_t size_i,
        float *a, float *tau) {
        qr_base::test_mulqtkernel_single(size_i,a, tau) ;
    }

    static void
    testmulq2(int32_t size_i,
        float *a, float *tau) {
        qr_base::test_mulqtkernel_double(size_i,a, tau) ;
    }


};

struct QRbase_improved {
    constexpr static char const *name = "qr_base_improved";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j) {
        return 0;
    }

    static void
    run(int32_t size_i,
        float *a, float *tau) {
        qr_base_improved::launch_tiled_qr(size_i,a, tau) ;
    }

    static void
    run_mulqt(int32_t size_k,int32_t size_i, int32_t size_j, 
        float *a, float *tau, float *b) {
        qr_base_improved::launch_mult_qt(size_k,size_i,size_j,a, tau, b) ;
    }


    static void
    testqr(int32_t size_i,
        float *a, float *tau) {
        qr_base_improved::test_qrkernel_single(size_i,a, tau) ;
    }

    static void
    testqr2(int32_t size_i,
        float *a, float *tau) {
        qr_base_improved::test_qrkernel_double(size_i,a, tau) ;
    }
    static void
    testmulq(int32_t size_i,
        float *a, float *tau) {
        qr_base_improved::test_mulqtkernel_single(size_i,a, tau) ;
    }

    static void
    testmulq2(int32_t size_i,
        float *a, float *tau) {
        qr_base_improved::test_mulqtkernel_double(size_i,a, tau) ;
    }


};
*/

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
    //results.push_back(run_all_configs<QRbase>(phase, data, configs));
    //results.push_back(run_all_configs<QRbase_improved>(phase, data, configs));
    results.push_back(run_all_configs<QRmulti>(phase, data, configs));
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
            auto [size_i, size_j] = config;
            double tflop = 2.0 * size_i * size_j * 1e-12;
            double tflop_per_sec = tflop / (elapsed_ms * 1e-3);
            file << "    {\n";
            file << "      \"size_i\": " << size_i << ",\n";
            file << "      \"size_j\": " << size_j << ",\n";
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

BenchmarkResults get_cublas_results(Phase phase,
    TestData  &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{"cusolver"};
    if (phase!= Phase::WARMUP) {
        printf(" benchmarking cusolver\n");
    }
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
    
        auto const &a = data.a.at({size_i, size_j});
        auto  &ref = data.ref.at({size_i, size_j});
        float *a_gpu;
        float *a_gpu_temp;
        float *tau_gpu;
        CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&a_gpu_temp, size_i * size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&tau_gpu, size_i * size_j * sizeof(float)));

        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));
        // Setup cuSolver for QR
        cusolverDnHandle_t solver_handle;
        CHECK_CUSOLVER(cusolverDnCreate(&solver_handle));
        int* devInfo;
        CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Query working space for QR
        int lwork;
        CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(solver_handle, size_i, size_j, a_gpu, size_i, &lwork));
        float* workspace;
        CHECK_CUDA(cudaMalloc(&workspace, lwork * sizeof(float)));
        
        
        if (phase == Phase::BENCHMARK) {
            double elapsed_ms = benchmark_ms(
            [&]() {
                CUDA_CHECK(cudaMemcpy( a_gpu_temp, a.data(), size_i * size_j * sizeof(float),cudaMemcpyHostToDevice));
                
            },
            [&]() {
                CHECK_CUSOLVER(cusolverDnSgeqrf(solver_handle, size_i, size_j, 
                    a_gpu_temp, size_i, tau_gpu, workspace, lwork, devInfo));
            });
            results.elapsed_ms[{size_i, size_j}] = elapsed_ms;
        }

        CUDA_CHECK(cudaMemcpy( a_gpu_temp, a.data(), size_i * size_j * sizeof(float),cudaMemcpyHostToDevice));
        CHECK_CUBLAS(cublasSgeam(handle, CUBLAS_OP_T,  CUBLAS_OP_N, 
            size_i, size_j,  &alpha, a_gpu_temp,size_j, &beta, nullptr, size_i, a_gpu,  size_i));
        CHECK_CUSOLVER(cusolverDnSgeqrf(solver_handle, size_i, size_j, 
            a_gpu, size_i, tau_gpu, workspace, lwork, devInfo));
        CHECK_CUBLAS(cublasSgeam(handle, CUBLAS_OP_T,  CUBLAS_OP_N, 
            size_j, size_i,  &alpha, a_gpu,size_i, &beta, nullptr, size_j, a_gpu_temp,  size_j));
        CUDA_CHECK(cudaMemcpy( ref.data(),a_gpu_temp,  size_i * size_j * sizeof(float),cudaMemcpyDeviceToHost));

                
        auto const &x = data.x.at({size_i, size_j});
        auto &xref =data.xref.at({size_i, size_j});
        float *x_gpu;
        float *x_gpu_temp;
        CUDA_CHECK(cudaMalloc(&x_gpu_temp, size_i * size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&x_gpu, size_i * size_j * sizeof(float)));

        CHECK_CUSOLVER(cusolverDnSormqr_bufferSize( solver_handle, 
            CUBLAS_SIDE_LEFT, CUBLAS_OP_T, size_i,size_j, std::min(size_i,size_j),         
            a_gpu, size_i,  tau_gpu, x_gpu,  size_i,  &lwork));
        float* workspace2;
        CHECK_CUDA(cudaMalloc(&workspace2, lwork * sizeof(float)));

        if (phase == Phase::BENCHMARK) {
            double elapsed_ms = benchmark_ms(
            [&]() {
                CUDA_CHECK(cudaMemcpy( x_gpu_temp, x.data(), size_i * size_j * sizeof(float),cudaMemcpyHostToDevice));
                
            },
            [&]() {
                CHECK_CUSOLVER(cusolverDnSormqr(solver_handle, 
                    CUBLAS_SIDE_LEFT, CUBLAS_OP_T, size_i,size_j, std::min(size_i,size_j),         
                    a_gpu, size_i,  tau_gpu, x_gpu_temp,  size_i,
                    workspace2, lwork, devInfo));
            });
            results.elapsed_ms_mulx[{size_i, size_j}] = elapsed_ms;
        }

        CUDA_CHECK(cudaMemcpy( x_gpu_temp, x.data(), size_i * size_j * sizeof(float),cudaMemcpyHostToDevice));
        CHECK_CUBLAS(cublasSgeam(handle, CUBLAS_OP_T,  CUBLAS_OP_N, 
            size_i, size_j,  &alpha, x_gpu_temp, size_j, &beta, nullptr, size_i, x_gpu,  size_i));
        
        // Apply Qt to the tile
        CHECK_CUSOLVER(cusolverDnSormqr(solver_handle, 
            CUBLAS_SIDE_LEFT, CUBLAS_OP_T, size_i,size_j, std::min(size_i,size_j),         
            a_gpu, size_i,  tau_gpu, x_gpu,  size_i,
            workspace2, lwork, devInfo));
        CHECK_CUBLAS(cublasSgeam(handle, CUBLAS_OP_T,  CUBLAS_OP_N, 
            size_j, size_i,  &alpha, x_gpu,size_i, &beta, nullptr,size_j,  x_gpu_temp,  size_j));
        CUDA_CHECK(cudaMemcpy( xref.data(), x_gpu_temp,size_i * size_j* sizeof(float),cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(a_gpu));
        CUDA_CHECK(cudaFree(a_gpu_temp));
        CUDA_CHECK(cudaFree(tau_gpu));
        CUDA_CHECK(cudaFree(workspace));
        CUDA_CHECK(cudaFree(workspace2));
        CUDA_CHECK(cudaFree(x_gpu));
        CUDA_CHECK(cudaFree(x_gpu_temp));
        CHECK_CUDA(cudaFree(devInfo));
        CHECK_CUSOLVER(cusolverDnDestroy(solver_handle));
    }
    return results;


}

BenchmarkResults get_cublas_results_extended(Phase phase,
    TestData  &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{"cusolver_ext"};
    if (phase!= Phase::WARMUP) {
        printf(" benchmarking cusolver extended\n");
    }
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
    
        auto const &a = data.a.at({size_i, size_j});
        float *a_gpu;
        float *tau_gpu;
        CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&tau_gpu, size_i * size_j * sizeof(float)));

        cusolverDnHandle_t solver_handle;
        CHECK_CUSOLVER(cusolverDnCreate(&solver_handle));
        int* devInfo;
        CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));

        
        
            double elapsed_ms = benchmark_ms(
            [&]() {
                CUDA_CHECK(cudaMemcpy( a_gpu, a.data(), size_i * size_j * sizeof(float),cudaMemcpyHostToDevice));
            },
            [&]() {
                int lwork;
                CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(solver_handle, size_i, size_j, a_gpu, size_i, &lwork));
                float* workspace;
                CHECK_CUDA(cudaMalloc(&workspace, lwork * sizeof(float)));
                CHECK_CUSOLVER(cusolverDnSgeqrf(solver_handle, size_i, size_j, 
                    a_gpu, size_i, tau_gpu, workspace, lwork, devInfo));
                    CUDA_CHECK(cudaFree(workspace));
            });
            results.elapsed_ms[{size_i, size_j}] = elapsed_ms;
            
        
                
        auto const &x = data.x.at({size_i, size_j});
        float *x_gpu;
        CUDA_CHECK(cudaMalloc(&x_gpu, size_i * size_j * sizeof(float)));

 
            elapsed_ms = benchmark_ms(
            [&]() {
                CUDA_CHECK(cudaMemcpy( x_gpu, x.data(), size_i * size_j * sizeof(float),cudaMemcpyHostToDevice));
            },
            [&]() {
                int lwork;
                CHECK_CUSOLVER(cusolverDnSormqr_bufferSize( solver_handle, 
                    CUBLAS_SIDE_LEFT, CUBLAS_OP_T, size_i,size_j, std::min(size_i,size_j),         
                    a_gpu, size_i,  tau_gpu, x_gpu,  size_i,  &lwork));
                float* workspace2;
                CHECK_CUDA(cudaMalloc(&workspace2, lwork * sizeof(float)));
                CHECK_CUSOLVER(cusolverDnSormqr(solver_handle, 
                    CUBLAS_SIDE_LEFT, CUBLAS_OP_T, size_i,size_j, std::min(size_i,size_j),         
                    a_gpu, size_i,  tau_gpu, x_gpu,  size_i,
                    workspace2, lwork, devInfo));
                CUDA_CHECK(cudaFree(workspace2));
            });
            results.elapsed_ms_mulx[{size_i, size_j}] = elapsed_ms;
        

        CUDA_CHECK(cudaFree(a_gpu));
        CUDA_CHECK(cudaFree(tau_gpu));
        CUDA_CHECK(cudaFree(x_gpu));
        CHECK_CUDA(cudaFree(devInfo));
        CHECK_CUSOLVER(cusolverDnDestroy(solver_handle));
    }
    return results;


}




void print_speedup(
    std::vector<BenchmarkConfig> const &configs,
    BenchmarkResults const &first,
    BenchmarkResults const &curesults, BenchmarkResults const &curesults2) {
    printf("\n Percentage of cublas %s :\n\n", first.name);
    printf("  %-6s  %-6s %-6s %-9s %-7s %-9s %-14s %-6s  %-9s %-7s %-9s %-14s\n", "size_i", "size_j", " type ", "time(ms) ", "vs cusolver","cutime(ms)", "cutime_ext(ms)",
     " type ", "time(ms) ","vs cusolver" ,"cutime(ms)", "cutime_ext(ms)");
    printf("  %-6s  %-6s %-6s  %-9s %-7s %-9s %-14s %-6s  %-9s %-7s %-9s %-14s\n", "------", "------", "------", "---------","-----------", "---------","--------------",
       "------",  "---------","-----------","---------" ,"--------------");
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        printf("  %6d  %6d ", size_i, size_j);
        auto it_first = first.elapsed_ms.find({size_i, size_j});
        auto it_cublas = curesults.elapsed_ms.find({size_i, size_j});
        auto it_cublas2 = curesults2.elapsed_ms.find({size_i, size_j});
        if (it_first != first.elapsed_ms.end() && it_cublas != curesults.elapsed_ms.end() && it_cublas2 != curesults2.elapsed_ms.end()) {
            printf(" QR  %8.02f  %8.02f %%  %8.02f %14.02f", it_first->second ,it_cublas->second/it_first->second *100, it_cublas->second, it_cublas2->second );
        } 
        it_first = first.elapsed_ms_mulx.find({size_i, size_j});
        it_cublas = curesults.elapsed_ms_mulx.find({size_i, size_j});
        it_cublas2 = curesults2.elapsed_ms_mulx.find({size_i, size_j});
        if (it_first != first.elapsed_ms_mulx.end() && it_cublas != curesults.elapsed_ms_mulx.end()  && it_cublas2 != curesults2.elapsed_ms_mulx.end()) {
            printf("      Qmul  %8.02f  %8.02f %%  %8.02f %14.02f", it_first->second ,it_cublas->second/it_first->second *100,  it_cublas->second, it_cublas2->second);
        } 
        printf("\n");
    }
}

int main(int argc, char **argv) {
    std::string test_data_dir = ".";
    if (char *c_str_test_data_dir = std::getenv("QR_TEST_DATA_DIR")) {
        test_data_dir = c_str_test_data_dir;
    }

    auto configs_test = std::vector<BenchmarkConfig>{
        {{tilesize,tilesize}},// {tilesize,2*tilesize},  {tilesize*2,tilesize}, {tilesize*2,tilesize*2}},// ,{128,128} },
    };
    auto configs = std::vector<BenchmarkConfig>{
        {{tilesize,tilesize},{128,128},{512,512},{1024,1024},{2048,2048},{4096,4096}},
    };

    
    auto data = read_test_data(Phase::TEST, test_data_dir, configs_test);
    get_cublas_results(Phase::WARMUP, data,configs_test);
    run_all_impls(Phase::TEST, data, configs_test);
    /*
    data = read_test_data(Phase::BENCHMARK, test_data_dir, configs);
    get_cublas_results(Phase::WARMUP, data,configs);
    auto curesults= get_cublas_results(Phase::BENCHMARK, data,configs);
    get_cublas_results_extended(Phase::WARMUP, data,configs);
    auto curesults_ext= get_cublas_results(Phase::BENCHMARK, data,configs);
    
    run_all_impls(Phase::WARMUP, data, configs);
    auto results = run_all_impls(Phase::BENCHMARK, data, configs);
    
    for (int32_t j = 0; j < results.size(); ++j) {
            print_speedup(configs, results.at(j),  curesults, curesults_ext);
    }*/

            
    //write_json_results("out/results.json", results);
    return 0;
}
