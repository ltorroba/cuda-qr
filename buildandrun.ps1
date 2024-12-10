cp src\gen_test_data.py build\
cp src\qr_kernels_orig.cu build\
cd build
py gen_test_data.py
nvcc  --std=c++17  --use_fast_math  --expt-relaxed-constexpr -O3  -arch=sm_89 -o qr qr_kernels_orig.cu -lcublas -lcurand -lcusolver
.\qr.exe