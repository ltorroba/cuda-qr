# cuda-qr

Make sure that G++10 is installed
```
sudo apt install g++-10
```

## Inference

To test the QRx and QRX kernels directly, run:
```
cd inference
nvcc -arch=sm_80 -ccbin=g++-10 -std=c++20 benchmark.cu reference_kernels.cu -o benchmark -lcublas -lcurand -lcusolver
./benchmark --verbose   # --verbose is optional
```

### QR kernels source code

The QR kernels sorurce code can be found in /final_project_devctr/src/qr_kernels.cu and is designed after https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.13010. The base implementation is copied from https://github.com/evelyne-ringoot/Avocado-sandwich/tree/main/KAbasedSVD/src and the reference implementation is cusolver cusolverDnDgeqrf.


### Docker build

To test the docker build (based on https://github.com/accelerated-computing-class/final_project_devctr):
Execute once:
```
./devtool build_devctr
```
Execute every change into src files:
```
./devtool build_project
py <path_to_telerun.py> submit build.tar
```

This will build a docker environment according to the /devctr/Dockerfile, build a tar file according to /src/build.sh and execute the /src/run.sh on telerun.

## To run project locally

```
cd src
./build_local.sh
cd build
../run.sh
```
