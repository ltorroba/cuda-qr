#include "reference_kernels.cuh"

__global__ void launch_QRx(const float* Q, const float* R, const float* x, float* output, 
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

__global__ void launch_QRX(const float* Q, const float* R, const float* X, float* output,
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
        sum += Qx * X[j + col * n];
    }
    output[row + col * m] = sum;
} 