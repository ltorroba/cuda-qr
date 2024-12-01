#pragma once

void reference_applyQt(int size_in, int diag_iter, const float* tau, float* matrix);

void launch_base_applyQt_singletile(int size_in, int diag_iter, const float* tau, float* out);

void launch_base_applyQt_singletile_evelyne(int size_in, int diag_iter, const float* tau, float* out);