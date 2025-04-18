#include <c10/core/ScalarType.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cute/tensor.hpp>
using cute::Layout, cute::Shape, cute::Stride, cute::Int;

template <typename scalar_t, int BLOCKSIZE_X, int BLOCKSIZE_Y>
__global__ void l2_normalize_rowmajor_kernel(cute::Tensor<scalar_t const *, Layout<Shape<int, int>, Stride<int, Int<1>>>> g_input,
                                             cute::Tensor<scalar_t *, Layout<Shape<int, int>, Stride<int, Int<1>>>> g_output,
                                             float *g_total_sum_of_squares, float epsilon) {

  using namespace cute;

  using accum_t = float;
  using BlockShape = Shape<_<BLOCKSIZE_Y>, _<BLOCKSIZE_X>>;

  using SMemLayout = Layout<BlockShape, Stride<Int<BLOCKSIZE_X + 1>, cute::_1>>;

  extern __shared__ char smem_buffer[];
  Tensor s_input = make_tensor(make_smem_ptr(reinterpret_cast<scalar_t *>(smem_buffer)), SMemLayout{});

  int block_idx_x = blockIdx.x;
  int block_idx_y = blockIdx.y;
  int thread_idx_x = threadIdx.x;
  int thread_idx_y = threadIdx.y;

  int tid = thread_idx_y * blockDim.x + thread_idx_x;
  int block_dim = blockDim.x * blockDim.y;

  Layout thread_layout_in_block = local_tile(BlockShape{}, Layout<Shape<Int<BLOCKSIZE_Y>, Int<BLOCKSIZE_X>>>{}, _1{}, tid);
  Tensor thread_local_input = local_partition(s_input, thread_layout_in_block, tid);

  Tensor g_input_block = logical_divide(g_input, BlockShape{}, Block{block_idx_y, block_idx_x});
  Tensor g_output_block = logical_divide(g_output, BlockShape{}, Block{block_idx_y, block_idx_x});

  // Copy Global -> Shared -------------------------------------------------------------------------
  using GMemLayoutAtom = Layout<Shape<_8, _16>>;
  using GMemCopyAtom = Copy_Atom<UniversalCopy<uint128_t>, scalar_t>;
  auto tiled_copy_G2S = make_tiled_copy(GMemCopyAtom{}, Layout<Shape<Int<BLOCKSIZE_Y>, Int<BLOCKSIZE_X>>>{}, select<0, 1>(thread_layout_in_block));

  copy(tiled_copy_G2S, g_input_block, s_input);
  __syncthreads();

  // Calculate Partial Sums (intrablock) -----------------------------------------------------------
  accum_t thread_sum_of_square = static_cast<accum_t>(0);
  for (auto i = 0; i < size(thread_local_input); ++i) {
    accum_t val = static_cast<accum_t>(thread_local_input(i));
    thread_sum_of_square += val * val;
  }

  // Blockwide reduction ---------------------------------------------------------------------------
  __shared__ accum_t s_partial_sums[BLOCKSIZE_X * BLOCKSIZE_Y];
  s_partial_sums[tid] = thread_sum_of_square;
  __syncthreads();

  for (auto offset = block_dim / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      s_partial_sums[tid] += s_partial_sums[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(g_total_sum_of_squares, s_partial_sums[0]);
  }
  __syncthreads();

  // Calculate normalization factor ----------------------------------------------------------------
  __shared__ accum_t s_inv_norm;
  if (tid == 0) {
    accum_t total_sum_of_squares = *g_total_sum_of_squares;
    s_inv_norm = rsqrtf(total_sum_of_squares + epsilon);
  }
  __syncthreads();

  // Apply normalization factor --------------------------------------------------------------------
  Tensor thread_local_output = local_partition(x_output_block, thread_layout_in_block, tid);
  for (auto i = 0; i < size(thread_local_input); ++i) {
    thread_local_output(i) = static_cast<scalar_t>(static_cast<accum_t>(thread_local_input(i)) * s_inv_norm);
  }
}
