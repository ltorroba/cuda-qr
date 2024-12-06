cp src\gen_test_data.py build\
cp src\qr_kernels.cu build\
cd build
py gen_test_data.py
nvcc  --std=c++17 --expt-relaxed-constexpr -O3 -arch=sm_86 -o qr qr_kernels.cu -lcublas -lcurand -lcusolver
.\qr.exe