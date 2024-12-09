#pragma once

void reference_applyQt(int size_in, int diag_iter, const float* tau, float* matrix);

void* reference_applyQt_fast_preamble(int size_in, int diag_iter, const float* tau, float* matrix);
void reference_applyQt_fast(int size_in, int diag_iter, const float* tau, float* matrix, void* workspace);
void reference_applyQt_fast_postamble(int size_in, int diag_iter, const float* tau, float* matrix, void* workspace);

void launch_base_applyQt_singletile(int size_in, int diag_iter, const float* tau, float* out);
void launch_base_applyQt_singletile_tc(int size_in, int diag_iter, const float* tau, float* out);

void launch_base_applyQt_singletile_evelyne(int size_in, int diag_iter, const float* tau, float* out);
void launch_base_applyQt_singletile_evelyne_2(int size_in, int diag_iter, const float* tau, float* out);