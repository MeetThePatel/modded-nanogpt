#include <ATen/native/cuda/Math.cuh>

constexpr float NS_a = 3.4445f;
constexpr float NS_b = -4.7750f;
constexpr float NS_c = 2.0315f;

template<typename scalar_t>
__inline__ __device__ scalar_t
warp_reduce_sum(scalar_t val)
{
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset)
  }
  return val
}

template<typename scalar_t>
__global__ void
fused_newton_schulz_kernel(const scalar_t* __restrict__ G,
                           scalar_t* __restrict__ output,
                           int M,
                           int N,
                           int ns_steps)
{
  // Will allocate SMEM for intermediate products.
  // [A (M x M), A^2 (M x M), B (M x M), X (M x N)]

  extern __shared__ scalar_t shared_buffers[];
  scalar_t* A = shared_buffers;
  scalar_t* A2 = A + (M * M);
  scalar_t* B = A2 + (M * M);
  scalar_t* X = B + (M * M);

  int total_numel = M * N;
  int tid = threadIdx.x;

  // Copy gradient tensor into X.
  for (auto idx = tid; idx < total_numel; idx += blockDim.x) {
    X[idx] = G[idx];
  }
  __syncthreads();

  // Normalization (by frob norm)
  __shared__ scalar_t frob_norm;

  scalar_t local_frob_norm = 0;
  for (auto idx = tid; idx < total_numel; idx += blockDim.x) {
    scalar_t val = X[idx];
    local_frob_norm += val * val;
  }
  local_frob_norm = warp_reduce_sum(local_frob_norm);

  if (tid == 0) {
    frob_norm = at::native::cuda::sqrt<scalar_t>
  }
}