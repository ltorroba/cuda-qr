#include "reference_kernels.cuh"

// Kernel implementations
__global__ void compute_QRx(const float* Q, const float* R, const float* x, float* output, 
                           int m, int n, int r) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    
    float sum = 0.0f;
    for (int j = 0; j < r; j++) {
        float Qx = 0.0f;
        for (int k = 0; k < n; k++) {
            Qx += Q[row + k * m] * R[k + j * n];
        }
        sum += Qx * x[j];
    }
    output[row] = sum;
}

__global__ void compute_QRX(const float* Q, const float* R, const float* X, float* output,
                           int m, int n, int r, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= m || col >= k) return;
    
    float sum = 0.0f;
    for (int j = 0; j < r; j++) {
        float Qx = 0.0f;
        for (int i = 0; i < n; i++) {
            Qx += Q[row + i * m] * R[i + j * n];
        }
        sum += Qx * X[j + col * r];
    }
    output[row + col * m] = sum;
}

// Launch function implementations
void launch_QRx(const float* Q, const float* R, const float* x, float* output, 
                int m, int n, int r) {
    // Choose block size and compute grid size
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    compute_QRx<<<grid, block>>>(Q, R, x, output, m, n, r);
}

void launch_QRX(const float* Q, const float* R, const float* X, float* output,
                int m, int n, int r, int k) {
    // Choose block dimensions
    const int BLOCK_DIM_X = 16;
    const int BLOCK_DIM_Y = 16;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((m + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (k + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    
    // Launch kernel
    compute_QRX<<<grid, block>>>(Q, R, X, output, m, n, r, k);
}

// Existing cuBLAS implementations remain unchanged
void cublas_ABx(cublasHandle_t handle, 
                const float* A, const float* B, const float* x,
                float* output, float* temp,
                int m, int n, int r) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // First compute Bx = temp
    cublasSgemv(handle, CUBLAS_OP_N,
                n, r,                  // dimensions
                &alpha,
                B, n,                  // matrix B (n x r)
                x, 1,                  // vector x (r x 1)
                &beta,
                temp, 1);             // output temp (n x 1)

    // Then compute A(Bx) = output
    cublasSgemv(handle, CUBLAS_OP_N,
                m, n,                  // dimensions
                &alpha,
                A, m,                  // matrix A (m x n)
                temp, 1,               // vector temp (n x 1)
                &beta,
                output, 1);           // output (m x 1)
}

void cublas_ABX(cublasHandle_t handle,
                const float* A, const float* B, const float* X,
                float* output, float* temp,
                int m, int n, int r, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // First compute BX = temp
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, k, r,              // dimensions
                &alpha,
                B, n,                  // matrix B (n x r)
                X, r,                  // matrix X (r x k)
                &beta,
                temp, n);             // output temp (n x k)

    // Then compute A(BX) = output
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, k, n,              // dimensions
                &alpha,
                A, m,                  // matrix A (m x n)
                temp, n,               // matrix temp (n x k)
                &beta,
                output, m);           // output (m x k)
} 