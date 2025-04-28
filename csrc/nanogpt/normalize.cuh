#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

__host__ __device__ inline int div_ceil(int numerator, int denominator) {
  if (denominator == 0) {
    return 0;
  }
  return (numerator + denominator - 1) / denominator;
}

extern "C" {
void reduce_sum_squares_partial_kernel_launcher(const __nv_bfloat16 *input, float *partial_sums, int N, int grid_dim, int block_dim,
                                                size_t shared_mem_size, cudaStream_t stream);

void reduce_final_sum_kernel_launcher(const float *partial_sums, float *total_sum_squared_out, int N, int grid_dim, int block_dim,
                                      size_t shared_mem_size, cudaStream_t stream);

void scale_by_inv_norm_inplace_kernel_launcher(__nv_bfloat16 *input, const float *total_sum_squared_ptr, int N, int grid_dim, int block_dim,
                                               cudaStream_t stream);
}
