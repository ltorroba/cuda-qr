# cuda-qr

Make sure that G++10 is installed
```
sudo apt install g++-10
```

To run, use:
```
nvcc -ccbin=g++-10 -std=c++20 benchmark.cu reference_kernels.cu -o benchmark -lcublas -lcurand -lcusolver
./benchmark
```
