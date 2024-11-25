#pragma once

// Original kernel declaration
template <int tilesize, int numthreads>
__global__ void base_applyQt_singletile(int size_in, 
                                       int diag_iter, 
                                       const float* tau, 
                                       float* out);

// Reference implementation declaration
void reference_applyQt(int size_in, 
                      int diag_iter, 
                      const float* tau, 
                      float* matrix);

