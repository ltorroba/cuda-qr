#ifndef REFERENCE_KERNELS_CUH
#define REFERENCE_KERNELS_CUH

// Kernel declarations
__global__ void launch_QRx(const float* Q, const float* R, const float* x, float* output, 
                          int m, int n, int r);

__global__ void launch_QRX(const float* Q, const float* R, const float* X, float* output,
                          int m, int n, int r, int k);

#endif // REFERENCE_KERNELS_CUH 