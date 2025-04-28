#include <torch/extension.h>

#include <cmath>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void reduce_sum_squares_partial_kernel(const __nv_bfloat16 *__restrict__ input, float *__restrict__ partial_sums, int N) {
  constexpr int BLOCK_DIM = 256;
  constexpr int VEC_SIZE = 8;

  extern __shared__ float s_mem[];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  int tid = block.thread_rank();
  int block_idx = blockIdx.x;
  int grid_dim = gridDim.x;

  float thread_sum_squared = 0.0f;

  int n_vec_elements = N / VEC_SIZE;
  int start_idx_vec = block_idx * BLOCK_DIM * VEC_SIZE + tid * VEC_SIZE;
  int stride_vec = grid_dim * BLOCK_DIM * VEC_SIZE;

  const uint4 *input_vec = reinterpret_cast<const uint4 *>(input);
  for (int i_vec = start_idx_vec / VEC_SIZE; i_vec < n_vec_elements; i_vec += stride_vec / VEC_SIZE) {
    uint4 loaded_data = input_vec[i_vec];
    const __nv_bfloat16 *vals = reinterpret_cast<const __nv_bfloat16 *>(&loaded_data);

#pragma unroll
    for (int k = 0; k < VEC_SIZE; ++k) {
      float val_f32 = __bfloat162float(vals[k]);
      thread_sum_squared += val_f32 * val_f32;
    }
  }

  int remaining_start_idx = n_vec_elements * VEC_SIZE;
  int start_idx_scalar = remaining_start_idx + block_idx * BLOCK_DIM + tid;
  int stride_scalar = grid_dim * BLOCK_DIM;
  for (int i = start_idx_scalar; i < N; i += stride_scalar) {
    float val_f32 = __bfloat162float(input[i]);
    thread_sum_squared += val_f32 * val_f32;
  }

  float warp_sum_squared = thread_sum_squared;
#pragma unroll
  for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
    warp_sum_squared += warp.shfl_down(warp_sum_squared, offset);
  }
  if (warp.thread_rank() == 0) {
    s_mem[warp.meta_group_rank()] = warp_sum_squared;
  }
  block.sync();

  float block_sum_squared = 0.0f;
  if (warp.meta_group_rank() == 0) {
    block_sum_squared = (tid < block.size() / warp.size()) ? s_mem[tid] : 0.0f;
#pragma unroll
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
      block_sum_squared += warp.shfl_down(block_sum_squared, offset);
    }
  }

  if (tid == 0) {
    partial_sums[block_idx] = block_sum_squared;
  }
}

__global__ void reduce_final_sum_kernel(const float *__restrict__ partial_sums, float *__restrict__ total_sum_squared_out, int N) {
  extern __shared__ float s_mem[];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  int tid = block.thread_rank();

  float thread_sum = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    thread_sum += partial_sums[i];
  }

  float warp_sum = thread_sum;
#pragma unroll
  for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
    warp_sum += warp.shfl_down(warp_sum, offset);
  }

  if (warp.thread_rank() == 0 && warp.meta_group_rank() < (blockDim.x / warp.size())) {
    s_mem[warp.meta_group_rank()] = warp_sum;
  }
  block.sync();

  float final_sum = 0.0f;
  if (warp.meta_group_rank() == 0) {
    final_sum = (tid < blockDim.x / warp.size()) ? s_mem[tid] : 0.0f;
#pragma unroll
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
      final_sum += warp.shfl_down(final_sum, offset);
    }
  }

  if (tid == 0) {
    *total_sum_squared_out = final_sum;
  }
}

__global__ void scale_by_inv_norm_inplace_kernel(__nv_bfloat16 *__restrict__ input, const float *__restrict__ total_sum_squared_ptr, int N) {
  constexpr int BLOCK_DIM = 256;
  constexpr int VEC_SIZE = 8;
  constexpr float epsilon = 1e-12f;

  __shared__ float s_inv_norm;

  if (threadIdx.x == 0) {
    float sum_squared = *total_sum_squared_ptr;
    s_inv_norm = (sum_squared > epsilon) ? rsqrtf(sum_squared) : 0.0f;
  }
  __syncthreads();

  float inv_norm = s_inv_norm;

  int tid = threadIdx.x;
  int block_idx = blockIdx.x;
  int grid_dim = gridDim.x;

  int n_vec_elements = N / VEC_SIZE;
  int start_idx_vec = block_idx * BLOCK_DIM * VEC_SIZE + tid * VEC_SIZE;
  int stride_vec = grid_dim * BLOCK_DIM * VEC_SIZE;

  uint4 *input_vec = reinterpret_cast<uint4 *>(input);
  for (int i_vec = start_idx_vec / VEC_SIZE; i_vec < n_vec_elements; i_vec += stride_vec / VEC_SIZE) {
    uint4 loaded_data = input_vec[i_vec];
    const __nv_bfloat16 *in_vals = reinterpret_cast<const __nv_bfloat16 *>(&loaded_data);
    __nv_bfloat16 out_vals[VEC_SIZE];

#pragma unroll
    for (int k = 0; k < VEC_SIZE; ++k) {
      float val_f32 = __bfloat162float(in_vals[k]);
      float scaled_val = val_f32 * inv_norm;
      out_vals[k] = __float2bfloat16(scaled_val);
    }
    input_vec[i_vec] = *reinterpret_cast<uint4 *>(out_vals);
  }

  int remaining_start_idx = n_vec_elements * VEC_SIZE;
  int start_idx_scalar = remaining_start_idx + block_idx * BLOCK_DIM + tid;
  int stride_scalar = grid_dim * BLOCK_DIM;
  for (int i = start_idx_scalar; i < N; i += stride_scalar) {
    float val_f32 = __bfloat162float(input[i]);
    float scaled_val = val_f32 * inv_norm;
    input[i] = __float2bfloat16(scaled_val);
  }
}

extern "C" {
void reduce_sum_squares_partial_kernel_launcher(const __nv_bfloat16 *input, float *partial_sums, int N, int grid_dim, int block_dim,
                                                size_t shared_mem_size, cudaStream_t stream) {
  reduce_sum_squares_partial_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(input, partial_sums, N);
}

void reduce_final_sum_kernel_launcher(const float *partial_sums, float *total_sum_squared_out, int N, int grid_dim, int block_dim,
                                      size_t shared_mem_size, cudaStream_t stream) {
  reduce_final_sum_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(partial_sums, total_sum_squared_out, N);
}

void scale_by_inv_norm_inplace_kernel_launcher(__nv_bfloat16 *input, const float *total_sum_squared_ptr, int N, int grid_dim, int block_dim,
                                               cudaStream_t stream) {
  scale_by_inv_norm_inplace_kernel<<<grid_dim, block_dim, 0, stream>>>(input, total_sum_squared_ptr, N);
}
}
