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

////////////////////////////////////////////////////////////////////////////////



namespace qr_base {

#define tilesize 32
#define numthreads 4



//-------------------- multiply by Q kernels-----------------------------------

__global__ void base_applyQt_singletile( //aplies Qt (given by householder reflectors on diagonal tile k) to the remainder of the row
    int size_in,
    int diag_iter,
    float const *tau,
    float *out) {
    int g = blockIdx.x;
    int i = threadIdx.x;
    int j = threadIdx.x;
    __shared__ float outs[tilesize][tilesize];
    __shared__ float Qs[tilesize][tilesize];
    __shared__ float cache[tilesize][numthreads];
    
    
    for (int l=j;l<tilesize;l+=numthreads){
        outs[i][l]=out[(i+diag_iter*tilesize)*size_in+l+(g+diag_iter+1)*tilesize];
        Qs[i][l]=out[(i+diag_iter*tilesize)*size_in+l];
    }

    __syncthreads();

    for (int k=0;k<tilesize-1;k++){
        float tmp_sum = 0.0f;
        for (int l=k+j;l<tilesize;l+=numthreads){
            tmp_sum+= Qs[l][k]*outs[l][i];
        }
        cache[i][j]=tmp_sum;
        __syncthreads();
        tmp_sum=outs[k][i];
        for (int l=0;l<numthreads;l++){
            tmp_sum+=cache[i][l];
        }
        tmp_sum*=tau[k];
        for (int l=k+j+1;l<tilesize;l+=numthreads){
            outs[l][i]-=tmp_sum*Qs[l][k];
        }
        if (j==0){
            outs[k][i]-=tmp_sum;
        }
        __syncthreads();
    }

    for (int l=j;l<tilesize;l+=numthreads){
        out[(i+diag_iter*tilesize)*size_in+l+(g+diag_iter)*tilesize]=outs[i][l];
    }
}

__global__ void base_applyQt_doubletile( //aplies Qt (given by householder reflectors on the tile at row_idx below diag_idx) to the remainder of the row, and to the row of diag_idx
    int size_in,
    int diag_iter,
    int row_iter,
    float const *tau,
    float *out) {
    int g = blockIdx.x;
    int i = threadIdx.x;
    int j = threadIdx.x;
    __shared__ float outs[2*tilesize][tilesize];
    __shared__ float Qs[tilesize][tilesize];
    __shared__ float cache[tilesize][numthreads];
    
    
    for (int l=j;l<tilesize;l+=numthreads){
        outs[i][l]=out[(i+(diag_iter)*tilesize)*size_in+l+(g+diag_iter+1)*tilesize];
        outs[i+tilesize][l]=out[(i+(diag_iter+row_iter)*tilesize)*size_in+l+(g+diag_iter+1)*tilesize];
        Qs[i][l]=out[(i+(diag_iter+row_iter)*tilesize)*size_in+l];
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
        tmp_sum*=tau[k];
        if (j==0){
            outs[k][i]-=tmp_sum;
        }
        for (int l=j+1;l<tilesize;l+=numthreads){
            outs[l+tilesize][i]-=tmp_sum*Qs[l][k];
        }
        
        __syncthreads();
    }

    for (int l=j;l<tilesize;l+=numthreads){
        out[(i+(diag_iter)*tilesize)*size_in+l+(g+diag_iter+1)*tilesize]=outs[i][l];
        out[(i+(diag_iter+row_iter)*tilesize)*size_in+l+(g+diag_iter+1)*tilesize]=outs[i+tilesize][l];
    }
}
//-------------------- calculate QR kernels-----------------------------------


__global__ void base_calcQR_singletile( //calculates in-place QR of diagonal tile
    int size_in,
    int diag_iter,
    float *tau,
    float *out) {
    int i = threadIdx.x;
    int j = threadIdx.x;
    __shared__ float outs[tilesize][tilesize];
    __shared__ float cache[tilesize];
    __shared__ float tauvals[2];

    outs[i][j]=out[(i+diag_iter*tilesize)*size_in+j];

    for(int iter=0;iter<N-1;iter++){
        if (i>iter && j==iter){
            cache[i]=outs[i][iter]*outs[i][iter];
        }
        __syncthreads();
        if (i==0 && j==0){
            float tmp_sum=0.0f;
            for (int l=iter;l<tilesize;l++){
                tmp_sum+=cache[l];
            }
            tmp_sum2=sqrt(tmp_sum+outs[iter][iter]*outs[iter][iter]);
            newvalue=outs[iter][iter];
            if (newvalue>0){
                newvalue+=tmp_sum2;
            }else{
                newvalue-=tmp_sum2;
            }
            tmp_sum2=sqrt(tmp_sum+newvalue*newvalue);
            tauvals[0]=2 * (newvalue/tmp_sum2)*(newvalue/tmp_sum2);
            tauvals[1]= newvalue;
            tau[iter]=tauvals[0];
        }
        if (j>=iter && i>=iter){
            float tmp_sum=0.0f;
            for (int k=iter;k>tilesize;k++){
                tmp_sum+=outs[k][iter]*outs[k][j];
            }
        }
        float tileiterj=outs[iter][j];
        float tileiteri = tile[i][iter];
        __syncthreads();
        if (j>=iter && i>=iter){
            float tmp_sum = (tmp_sum / tauvals[1]+tileiterj)*tauvals[0];
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
        out[(i+diag_iter*tilesize)*size_in+j]=outs[i][j];

    }

}

__global__ void base_calcQR_singletile( //calculates in-place QR of diagonal tile combined with row_idx tile below
    int size_in,
    int diag_iter,
    int row_iter,
    float *tau,
    float *out) {
    int i = threadIdx.x;
    int j = threadIdx.x;
    __shared__ float outs[2*tilesize][tilesize];
    __shared__ float cache[2*tilesize];
    __shared__ float tauvals[2];

    outs[i][j]=out[(i+diag_iter*tilesize)*size_in+j];
    outs[i+tilesize][j]=out[(i+(diag_iter+row_iter)*tilesize)*size_in+j];

    for(int iter=0;iter<N;iter++){
        if (j==iter){
            cache[i]=outs[i+tilesize][iter]*outs[i+tilesize][iter];
        }
        __syncthreads();
        if (i==0 && j==0){
            float tmp_sum=0.0f;
            for (int l=0;l<tilesize;l++){
                tmp_sum+=cache[l];
            }
            tmp_sum2=sqrt(tmp_sum+outs[iter][iter]*outs[iter][iter]);
            newvalue=outs[iter][iter];
            if (newvalue>0){
                newvalue+=tmp_sum2;
            }else{
                newvalue-=tmp_sum2;
            }
            tmp_sum2=sqrt(tmp_sum+newvalue*newvalue);
            tauvals[0]=2 * (newvalue/tmp_sum2)*(newvalue/tmp_sum2);
            tauvals[1]= newvalue;
            tau[iter]=tauvals[0];
        }
        float tileiterj=outs[iter][j];
        float tileiiteri = tile[i+tilesize][iter];
        if (j>=iter){
            float tmp_sum=0.0f;
            for (int k=tilesize;k>tilesize*2;k++){
                tmp_sum+=outs[k][iter]*outs[k][j];
            }
        }
        __syncthreads();
        if (j>=iter){
            float tmp_sum = ( tauvals[1]*tileiterj);
            if (i==iter){
                outs[i][j]-=tauiterj*tauvals[0];
            }
            if(j>iter){
                outs[i+tilesize][j]-=tileiiter*tmp_sum * tauvals[0]/tauvals[1]/tauvals[1];
            }

        }
        if (j==0){
            outs[i+tilesize][iter]=tileiiter / tauvals[1];
        }
        __syncthreads();
        out[(i+diag_iter*tilesize)*size_in+j]=outs[i][j];
        out[(i+(diag_iter+row_iter)*tilesize)*size_in+j]=outs[i+tilesize][j];


    }

}



        

void launch_tiled_qr(
    int32_t size_i,
    float *a, float *tau) {
        /*
    if ( numthreadsperblockb %4 !=0 || nummemperblock %4!=0 || ilpnuma%4!=0 || ilpnumb%4!=0){
            throw std::invalid_argument( "Not implemented for this argument size" );
    }
    int result=0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor (&result,matmul_improved_macro2, numthreadsperblocka*numthreadsperblockb, 0);
    printf("%d\n",result);
    //uint32_t shmem_size_bytes = (((numthreadsperblocka)*(nummemperblock+1)*ilpnuma+(numthreadsperblockb)*(nummemperblock)*ilpnumb));
    int32_t noblocksa=(size_i+numthreadsperblocka*ilpnuma-1)/(numthreadsperblocka*ilpnuma);
    int32_t noblocksb=((size_j+numthreadsperblockb*ilpnumb-1))/(numthreadsperblockb*ilpnumb);
    dim3 num_blocks = dim3(noblocksa*noblocksb,1,1  );
    dim3 block_size = dim3(numthreadsperblocka*numthreadsperblockb,1,1);
    //CUDA_CHECK(cudaFuncSetAttribute( matmul_improved_macro2, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_bytes));
    matmul_improved_macro2<<<num_blocks, block_size>>>(size_i,size_j,size_k,noblocksa,noblocksb,a,b,c);
        */
    }

    void test_qrkernel_single(
        int32_t size_i,
        float *a, float *tau) {

        base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,0,*tau,out); 
    
    
        }

    void test_mulqtkernel_single(
        int32_t size_i,
        float *a, float *tau) {

            base_calcQR_singletile<<<1,dim3(tilesize,tilesize)>>>(size_i,0,tau,out); 
            base_applyQt_singletile<<<1,dim3(tilesize,numthreads)>>>(size_i,0,tau,out); 


    
        }

        


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
};

TestData read_test_data(
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

        if (data.ref.find({size_i, size_j, size_k}) == data.ref.end()) {
            data.ref[{size_i, size_j, size_k}] = read_data(
                path_prefix + "ref_" + std::to_string(size_i) + "_" +
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
    float *workspace_gpu = nullptr;
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
    CUDA_CHECK(cudaFree(workspace_gpu));
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



struct QRbase {
    constexpr static char const *name = "qr_base";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        float *a) {
        qr_base::launch_tiled_qr(size_i,a, tau) ;
    }

    static void
    testqr(int32_t size_i,
        float *a, float *tau) {
        qr_base::test_qrkernel_single(size_i,a, tau) ;
    }

    static void
    testmulq(int32_t size_i,
        float *a, float *tau) {
        qr_base::test_mulqtkernel_single(size_i,a, tau) ;
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
    if (char *c_str_test_data_dir = std::getenv("QR_TEST_DATA_DIR")) {
        test_data_dir = c_str_test_data_dir;
    }

    auto configs = std::vector<BenchmarkConfig>{
        {2048},
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
