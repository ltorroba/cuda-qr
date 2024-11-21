#!/bin/bash

# Output build directory path.
CTR_BUILD_DIR=/build

echo "Building the project..."

# ----------------------------------------------------------------------------------
# ------------------------ PUT YOUR BULDING COMMAND(s) HERE ------------------------
# ----------------------------------------------------------------------------------
# ----- This sctipt is executed inside the development container:
# -----     * the current workdir contains all files from your src/
# -----     * all output files (e.g. generated binaries, test inputs, etc.) must be places into $CTR_BUILD_DIR
# ----------------------------------------------------------------------------------
#

# Build code.
# we use --expt-relaxed-constexpr to avoid warning #20013-D (not the best solution)
nvcc --expt-relaxed-constexpr -O3 --use_fast_math -gencode arch=compute_86,code=[sm_86,compute_86] -o ${CTR_BUILD_DIR}/qr qr_kernels.cu -lcublas -lcurand -lcusolver

# Upload test generation script.
cp gen_test_data.py ${CTR_BUILD_DIR}/
