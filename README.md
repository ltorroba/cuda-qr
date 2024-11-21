# cuda-qr

Make sure that G++10 is installed
```
sudo apt install g++-10
```

## Inference

To test the QRx and QRX kernels directly, run:
```
cd inference
nvcc -ccbin=g++-10 -std=c++20 benchmark.cu reference_kernels.cu -o benchmark -lcublas -lcurand -lcusolver
./benchmark
```

### QR kernels source code

The QR kernels sorurce code can be found in /final_project_devctr/src/qr_kernels.cu and is designed after https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.13010. The base implementation is copied from https://github.com/evelyne-ringoot/Avocado-sandwich/tree/main/KAbasedSVD/src and the reference implementation is cusolver cusolverDnDgeqrf.


### Docker build

To test the docker build (based on https://github.com/accelerated-computing-class/final_project_devctr):

```
./final_project_devctr/devtool build_devctr
./final_project_devctr/devtool build_project
python3 <path_to_telerun.py> submit build/build.tar
```

This will build a docker environment according to the /final_project_devctr/Dockerfile, build a tar file according to /final_project_devctr/src/build.sh and execute the /final_project_devctr/src/run.sh on telerun.