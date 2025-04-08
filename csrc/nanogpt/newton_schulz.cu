#include <cute/tensor.hpp>

constexpr float a = 3.4445f;
constexpr float b = -4.7750f;
constexpr float c = 2.0315f;

__global__ void
compute_B(const float* A_d,
          const float* A2_d,
          float* B_d,
          int dim,
          float b,
          float c)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dim * dim;
  if (idx < total) {
    B_d[idx] = b * A_d[idx] + c * A2_d[idx];
  }
}

extern "C" __global__ void
newton_schulz_fused_kernel(const float* G_d, float* X_d, int rows, int cols)
{
  extern __shared__ float shared_mem[];
  float* A = shared_mem;
  float* A2 = A + (rows * rows);
  float* B = A2 + (rows * rows);
  float* temp = B + (rows * rows);

  int threadid = threadIdx.x;
  int num_threads = blockDim.x;

  for (int index = threadid; index < rows * cols; index += num_threads) {
    X_d[index] = G_d[index];
  }
  __syncthreads();
}