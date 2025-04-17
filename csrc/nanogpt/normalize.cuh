#pragma once

#include <cuda_runtime.h>

template <typename scalar_t>
void normalize_(scalar_t *__restrict__ X, // (M, N)
                const int M, const int N);
