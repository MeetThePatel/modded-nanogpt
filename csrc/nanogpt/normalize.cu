#include <c10/core/ScalarType.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Accumulation Type
template <typename scalar_t>
struct AccumType {
  using type = float;
};

template <>
struct AccumType<double> {
  using type = double;
};

// Epsilon
template <typename scalar_t>
struct Epsilon;

template <>
struct Epsilon<double> {
  static constexpr double value = 1e-8;
};

template <>
struct Epsilon<float> {
  static constexpr float value = 1e-8f;
};

template <typename accum_t>
__device__ inline accum_t warpReduceSum(accum_t val) {
  static_assert(std::is_same_v<accum_t, float> ||
                    std::is_same_v<accum_t, double>,
                "warpReduceSum accum_t must be either float or double.");
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename scalar_t, typename accum_t, int BLOCK_SIZE>
__global__ void sum_of_squares_kernel(const scalar_t *__restrict__ X,
                                      const size_t total_numel,
                                      accum_t *__restrict__ partial_sums) {
  __shared__ accum_t sData[BLOCK_SIZE / 32];

  accum_t thread_sum = static_cast<accum_t>(0.0);
  for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_numel;
       idx += gridDim.x * blockDim.x) {
    accum_t val = static_cast<accum_t>(X[idx]);
    thread_sum += val * val;
  }

  accum_t warp_sum = warpReduceSum(thread_sum);

  if ((threadIdx.x) % 32 == 0) {
    sData[threadIdx.x / 32] = warp_sum;
  }

  __syncthreads();

  accum_t block_sum = static_cast<accum_t>(0.0);
  if (threadIdx.x < (blockDim.x / 32)) {
    block_sum = sData[threadIdx.x];
  }
  if (threadIdx.x < 32) {
    block_sum = warpReduceSum(block_sum);
  }
  if (threadIdx.x == 0) {
    partial_sums[blockIdx.x] = block_sum;
  }
}

template <typename accum_t, int BLOCK_SIZE>
__global__ void reduce_partials_kernel(const accum_t *__restrict__ partial_sums,
                                       const int num_partials,
                                       accum_t *__restrict__ final_sum) {
  __shared__ accum_t sData[BLOCK_SIZE];

  accum_t thread_sum = static_cast<accum_t>(0.0);
  for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_partials;
       idx += gridDim.x * blockDim.x) {
    thread_sum += partial_sums[idx];
  }

  sData[threadIdx.x] = thread_sum;
  __syncthreads();

#pragma unroll
  for (auto offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      sData[threadIdx.x] += sData[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    if (sData[0] != static_cast<accum_t>(0.0)) {
      atomicAdd(final_sum, sData[0]);
    }
  }
}

template <typename scalar_t, typename accum_t, int BLOCK_SIZE>
__global__ void scale_kernel(scalar_t *__restrict__ X, const size_t total_numel,
                             const accum_t *__restrict__ final_sum) {
  __shared__ accum_t inv_sqrt_sum_SHARED;

  if (threadIdx.x == 0) {
    accum_t total_sum = final_sum[0];
    accum_t sum_plus_eps = total_sum + Epsilon<accum_t>::value;
    accum_t inv_sqrt_sum_val;

    if constexpr (std::is_same_v<accum_t, float>) {
      inv_sqrt_sum_val = (sum_plus_eps <= 0.0f) ? 0.0f : rsqrt(sum_plus_eps);
    } else {
      inv_sqrt_sum_val = (sum_plus_eps <= 0.0) ? 0.0 : rsqrt(sum_plus_eps);
    }
    inv_sqrt_sum_SHARED = inv_sqrt_sum_val;
  }
  __syncthreads();

  accum_t inv_sqrt_sum = inv_sqrt_sum_SHARED;

  for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_numel;
       idx += gridDim.x * blockDim.x) {
    accum_t val = static_cast<accum_t>(X[idx]);
    val *= inv_sqrt_sum;
    X[idx] = static_cast<accum_t>(val);
  }
}

template <typename scalar_t>
void normalize_(scalar_t *__restrict__ X, // (M, N)
                const int M, const int N) {

  if (M <= 0 || N <= 0)
    return;

  const size_t total_numel = static_cast<size_t>(M) * N;
  using accum_t = typename AccumType<scalar_t>::type;

  int device;
  cudaGetDevice(&device);

  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

  // Compute sum of squares ----------------------------------------------------

  const int BLOCK_SIZE_1 = 256; // TODO: Tune this.

  int target_blocks_1 = sm_count * 32;
  int grid_size_1 = std::min(
      target_blocks_1, (int)((total_numel + BLOCK_SIZE_1 - 1) / BLOCK_SIZE_1));
  grid_size_1 = std::max(1, grid_size_1);

  accum_t *partial_sums = nullptr;
  accum_t *final_sum = nullptr;
  cudaMalloc(&partial_sums, grid_size_1 * sizeof(accum_t));
  cudaMalloc(&final_sum, sizeof(accum_t));

  cudaMemset(final_sum, 0, sizeof(accum_t));

  sum_of_squares_kernel<scalar_t, accum_t, BLOCK_SIZE_1>
      <<<grid_size_1, BLOCK_SIZE_1>>>(X, total_numel, partial_sums);

  // Reduce partial sums. ------------------------------------------------------

  const int BLOCK_SIZE_2 = 256; // TODO: Tune this.

  int target_blocks_2 = sm_count * 32;
  int grid_size_2 = std::min(
      target_blocks_2, (int)((total_numel + BLOCK_SIZE_2 - 1) / BLOCK_SIZE_2));
  grid_size_2 = std::max(1, grid_size_2);

  if (grid_size_2 > 0) {
    reduce_partials_kernel<accum_t, BLOCK_SIZE_2>
        <<<grid_size_2, BLOCK_SIZE_2>>>(partial_sums, grid_size_1, final_sum);
  }

  // Scale element-wise. -------------------------------------------------------
  const int BLOCK_SIZE_3 = 256; // TODO: Tune this.
  int target_blocks_3 = sm_count * 32;
  int grid_size_3 = std::min(
      target_blocks_3, (int)((total_numel + BLOCK_SIZE_3 - 1) / BLOCK_SIZE_3));
  grid_size_3 = std::max(1, grid_size_3);

  scale_kernel<scalar_t, accum_t, BLOCK_SIZE_3>
      <<<grid_size_3, BLOCK_SIZE_3>>>(X, total_numel, final_sum);

  cudaFree(partial_sums);
  cudaFree(final_sum);
}

#define DECLARE_NORMALIZE(type)                                                \
  template void normalize_<type>(type *__restrict__, int, int);
DECLARE_NORMALIZE(double)
DECLARE_NORMALIZE(float)
DECLARE_NORMALIZE(c10::BFloat16)
DECLARE_NORMALIZE(c10::Half)
