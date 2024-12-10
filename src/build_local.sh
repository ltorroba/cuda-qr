#!/bin/bash
python gen_test_data.py
nvcc  --std=c++17  --use_fast_math  --expt-relaxed-constexpr -O3  -gencode arch=compute_86,code=[sm_86,compute_86]  -o qr qr_kernels_orig.cu -lcublas -lcurand -lcusolver
./qr