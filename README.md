# cuda-qr

To run, use:

```
nvcc benchmark.cu reference_kernels.cu -o benchmark -lcublas -lcurand -lcusolver
./benchmark
```
