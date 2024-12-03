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



namespace qr_base {

#define tilesize 8
#define numthreads 4


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
            }else{
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
        if (j>=iter){ //j 0,1
            tmp_sum += ( tauvals[1]*tileiterj);
            if (i==iter){ //i 0
                outs[i][j]-=tmp_sum*tauvals[0]/tauvals[1];
            }
            if(j>iter){// i 1
                outs[i+tilesize][j]-=tileiiter*tmp_sum * tauvals[0]/tauvals[1]/tauvals[1];
            }
            
    

        }
        __syncthreads();
        if (j==0){ //j 0 i 0 1
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
        base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,iter,tau,a); 
        base_applyQt_singletile<<<nb_blocks-1-iter,dim3(tilesize,numthreads)>>>(size_i,size_i,iter,true,tau,a,a); 
        for (int row=1;row+iter<nb_blocks;row++){
            base_calcQR_doubletile<<<1,dim3(tilesize,tilesize)>>>(size_i,iter,row,tau,a);
            base_applyQt_doubletile<<<nb_blocks-1-iter,dim3(tilesize,numthreads)>>>(size_i,size_i,iter,row,true,tau,a,a); 
        }
    }
    base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,nb_blocks-1,tau,a); 
        
    }

    void launch_mult_qt(
        int32_t size_k, int32_t size_i,  int32_t size_j,  
        float *a, float *tau, float *b) {

        if ( size_k%tilesize !=0 ){
                throw std::invalid_argument( "Not implemented for this argument size" );
        }
        
        for(int iter=0;iter<min(size_i,size_j)/tilesize;iter++){
            base_applyQt_singletile<<<1,dim3(tilesize,numthreads)>>>(size_j,size_k,iter,false,tau,a, b); 
            for (int row=1;row+iter<(size_i/tilesize);row++){

                base_applyQt_doubletile<<<1,dim3(tilesize,numthreads)>>>(size_j,size_k,iter,row,false,tau,a, b); 
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

////////////////////////////////////////////////////////////////////////////////


enum class Phase {
    TEST,
    WARMUP,
    BENCHMARK,
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
};

struct TestData {
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> ref;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> x;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> xref;
};

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
                path_prefix + "ref_" + std::to_string(size_i) + "_" +
                    std::to_string(size_j) + ".bin",
                size_i * size_j);
        }
        if (data.x.find({size_i, size_j}) == data.x.end()) {
            data.x[{size_i, size_j}] = read_data(
                path_prefix + "x_" + std::to_string(size_i) +"_" +
                std::to_string(size_j) + ".bin",
                size_i * tilesize);
        }
        

        if (data.xref.find({size_i, size_j}) == data.xref.end()) {
            data.xref[{size_i, size_j}] = read_data(
                path_prefix + "xmul_" + std::to_string(size_i) +"_" +
                std::to_string(size_j) + ".bin",
                size_i * tilesize);
        }
        
            

    }
    return data;
}

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t, int32_t>, double> elapsed_ms;
    std::map<std::tuple<int32_t>, double> elapsed_ms_mulx;
};


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

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size_i * size_j * sizeof(float),
        cudaMemcpyHostToDevice));

    if (phase == Phase::BENCHMARK) {
        printf("  QR    %6d  %6d  ", size_i, size_j);
    } else {
        printf("  warmup      QR   %6d  %6d", size_i, size_j);
    }

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

    }else{
        Impl::run(size_j,   a_gpu, tau_gpu);
    }

    std::vector<float> c_out_host(size_i * size_j);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        a_gpu,
        size_i * size_j * sizeof(float),
        cudaMemcpyDeviceToHost));

    double mse = 0.0;
    for (int32_t i = 0; i < size_i; ++i) {
        for (int32_t j = i; j < size_j; ++j) {
            float diff = abs(abs(c_out_host[i * size_j + j]) - abs(ref[i * size_j + j]))/abs(ref[i * size_j + j]);
            mse+= diff*diff;
        }
    }
    
    float rel_rmse = std::sqrt(mse) / max(size_i,size_j);

    if (phase == Phase::BENCHMARK || phase == Phase::TEST ) {
        printf("   %8.02e", rel_rmse);
    }

    if ((rel_rmse > 1e-3 && size_i>=size_j) || rel_rmse!=rel_rmse ) {
        if (phase == Phase::BENCHMARK) {
            printf("  %9s  %7s", "-", "-");
        } else if (phase == Phase::TEST){
            printf("\n");
            printf("  expected output:\n");
            print_matrix(size_i, size_j, ref);
            printf("\n");
            printf("  obtained output:\n");
            print_matrix(size_i,  size_j, c_out_host);
        }
    } else if (phase == Phase::BENCHMARK){
        double target_time_ms = 200.0;
        double elapsed_ms = benchmark_ms(
            target_time_ms,
            4,
            [&]() {
                CUDA_CHECK(cudaMemcpy( a_gpu, a.data(), size_i * size_j * sizeof(float),cudaMemcpyHostToDevice));
            },
            [&]() {
                Impl::run(size_j,  a_gpu, tau_gpu);
            });

            printf(" %9.02f ", elapsed_ms);
            results.elapsed_ms[{size_i, size_j}] = elapsed_ms;

    }

    printf("\n");
    
    auto const &x = data.x.at({size_i, size_j});
    auto const &xref = data.xref.at({size_i, size_j});

    float *x_gpu;
    CUDA_CHECK(cudaMalloc(&x_gpu, size_i * tilesize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        x_gpu,
        x.data(),
        size_i * tilesize * sizeof(float),
        cudaMemcpyHostToDevice));

    if (phase == Phase::BENCHMARK) {
        printf("  Qmul  %6d  %6d  ", size_i, size_j);
    } else {
        printf("  warmup  mul Qt   %6d ", size_i);
    }

    Impl::run_mulqt(tilesize, size_i,size_j, a_gpu, tau_gpu, x_gpu);


    std::vector<float> x_out_host(size_i * tilesize * sizeof(float));
    CUDA_CHECK(cudaMemcpy(
        x_out_host.data(),
        x_gpu,
        size_i * tilesize* sizeof(float),
        cudaMemcpyDeviceToHost));
        
    mse = 0.0;
    for (int32_t i = 0; i < size_i; ++i) {
        for (int32_t j = 0; j < tilesize; ++j) {
            float diff = abs(abs(x_out_host[i * tilesize + j]) - abs(xref[i * tilesize + j]))/abs(xref[i * tilesize + j]);
            mse+= diff*diff;
        }
    }
    
    rel_rmse = std::sqrt(mse) / max(size_i,tilesize);
    if (size_i>=size_j){
        rel_rmse=-1;
    }
    
    if (phase == Phase::BENCHMARK  || phase == Phase::TEST  ) {
        printf("   %8.02e", rel_rmse);
    }

    if (rel_rmse > 1e-3 || rel_rmse!=rel_rmse) {
        if (phase == Phase::BENCHMARK) {
            printf("  %9s  %7s", "-", "-");
        } else if (phase == Phase::TEST){
            printf("\n");
            printf("  input :\n");
            print_matrix(size_i, tilesize, x);
            printf("\n");
            printf("  input a:\n");
            print_matrix(size_i, size_j, c_out_host);
            printf("\n");
            printf("  expected output:\n");
            print_matrix(size_i, tilesize, xref);
            printf("\n");
            printf("  obtained output:\n");
            print_matrix(size_i,  tilesize, x_out_host);
        }
    } else if (phase == Phase::BENCHMARK){
        double target_time_ms = 200.0;
        double elapsed_ms = benchmark_ms(
            target_time_ms,
            4,
            [&]() {
                CUDA_CHECK(cudaMemcpy( x_gpu, x.data(), size_i * tilesize * sizeof(float),cudaMemcpyHostToDevice));
            },
            [&]() {
                Impl::run_mulqt(tilesize, size_i,size_j, a_gpu, tau_gpu, x_gpu);
            });

            printf(" %9.02f ", elapsed_ms);
            results.elapsed_ms_mulx[{size_i}] = elapsed_ms;
        
    }

    printf("\n");
    
    
   


    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(tau_gpu));
}

template <typename Impl>
BenchmarkResults run_all_configs(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{Impl::name};
    if (phase == Phase::WARMUP) {
        printf("warmup %s:\n\n", Impl::name);
    } else if (phase == Phase::TEST){
        printf("testing %s:\n\n", Impl::name);
    }else {
        printf("%s:\n\n", Impl::name);
        printf(
            " %-6s  %-6s  %-6s   %-8s  %-9s  \n",
            " type ",
            "size_i",
            "size_j",
            "RRMSE",
            "time (ms)");
        printf(
            " %-6s  %-6s  %-6s  %-8s  %-9s\n",
            "------",
            "------",
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


std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
    results.push_back(run_all_configs<QRbase>(phase, data, configs));
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
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{"cusolver"};
    if (phase == Phase::WARMUP) {
        printf("warmup cusolver:\n\n");
    }else {
        printf(" benchmarking cusolver:\n\n");
        printf(
            "  %-6s  %-6s   %-9s  \n",
            "size_i",
            "size_j",
            "time (ms)");
        printf(
            " %-6s  %-6s   %-9s  \n",
            "------",
            "------",
            "---------");
    }
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
    
        auto const &a = data.a.at({size_i, size_j});
        float *a_gpu;
        float *tau_gpu;
        CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&tau_gpu, size_i * size_j * sizeof(float))); // TODO: determine size more accurately
    
        CUDA_CHECK(cudaMemcpy( a_gpu, a.data(), size_i * size_j * sizeof(float),cudaMemcpyHostToDevice));

        // Setup cuSolver for QR
        cusolverDnHandle_t solver_handle;
        CHECK_CUSOLVER(cusolverDnCreate(&solver_handle));
        
        // Query working space for QR
        int lwork;
        CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(solver_handle, size_j, size_j, a_gpu, size_j, &lwork));
        
        // Allocate working space
        float* workspace;
        CHECK_CUDA(cudaMalloc(&workspace, lwork * sizeof(float)));
        int* devInfo;
        CHECK_CUDA(cudaMalloc(&devInfo, sizeof(int)));
      
    
        if (phase == Phase::BENCHMARK) {
            printf("  %6d  %6d  ", size_i, size_j);
        } else {
            printf("  warmup %6d  %6d", size_i, size_j);
        }
    
        
        // Compute QR factorization in-place in AB
        CHECK_CUSOLVER(cusolverDnSgeqrf(solver_handle, size_i, size_j, 
            a_gpu, size_j, tau_gpu, workspace, lwork, devInfo));
    
        double target_time_ms = 200.0;
        double elapsed_ms = benchmark_ms(
            target_time_ms,
            4,
            [&]() {
                CUDA_CHECK(cudaMemcpy( a_gpu, a.data(), size_i * size_j * sizeof(float),cudaMemcpyHostToDevice));
                if (lwork > 0) {
                    CUDA_CHECK(cudaMemset(workspace, 0, lwork));
                }
            },
            [&]() {
                CHECK_CUSOLVER(cusolverDnSgeqrf(solver_handle, size_i, size_j, 
                    a_gpu, size_j, tau_gpu, workspace, lwork, devInfo));
            });

        if (phase == Phase::BENCHMARK) {

            results.elapsed_ms[{size_i, size_j}] = elapsed_ms;
            printf("  %9.02f ", elapsed_ms);
        }
        
    
        printf("\n");
    
        CUDA_CHECK(cudaFree(a_gpu));
        CUDA_CHECK(cudaFree(tau_gpu));
        CUDA_CHECK(cudaFree(workspace));
        CHECK_CUDA(cudaFree(devInfo));
        CHECK_CUSOLVER(cusolverDnDestroy(solver_handle));
    }
    printf("\n");
    return results;


}

void print_speedup(
    std::vector<BenchmarkConfig> const &configs,
    BenchmarkResults const &first,
    BenchmarkResults const &curesults) {
    printf("\n Percentage of cublas %s :\n\n", first.name);
    printf("  %-6s  %-6s  %-7s\n", "size_i", "size_j",  "vs cusolver");
    printf("  %-6s  %-6s  %-7s\n", "------", "------",   "-----------");
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        printf("  %6d  %6d ", size_i, size_j);
        auto it_first = first.elapsed_ms.find({size_i, size_j});
        auto it_cublas = curesults.elapsed_ms.find({size_i, size_j});
        if (it_first != first.elapsed_ms.end() && it_cublas != curesults.elapsed_ms.end()) {
            printf("  %6.02f %%", it_cublas->second/it_first->second *100 );
        } else {
            printf("  %7s", "-");
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
        {{tilesize,tilesize}, {tilesize,2*tilesize},  {tilesize*2,tilesize}, {tilesize*2,tilesize*2}},
    };
    auto configs = std::vector<BenchmarkConfig>{
        {{128,128},{512,512},{2048,2048}},
    };

    
    auto data = read_test_data(Phase::TEST, test_data_dir, configs_test);
    
    run_all_impls(Phase::TEST, data, configs_test);
    /*
    data = read_test_data(Phase::BENCHMARK, test_data_dir, configs);
    run_all_impls(Phase::WARMUP, data, configs);
    auto results = run_all_impls(Phase::BENCHMARK, data, configs);
    get_cublas_results(Phase::WARMUP, data,configs);
    auto curesults= get_cublas_results(Phase::BENCHMARK, data,configs);

    for (int32_t j = 0; j < results.size(); ++j) {
            print_speedup(configs, results.at(j),  curesults);
    }
    */
    
            
    //write_json_results("out/results.json", results);
    return 0;
}
